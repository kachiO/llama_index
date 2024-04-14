#!/usr/bin/env python3

import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
import unicodedata

import boto3  # type: ignore
from botocore.exceptions import ClientError
from unidecode import unidecode

from enum import Enum
import logging
import botocore.waiter


logger = logging.getLogger(__name__)

FORMATS = [".wav", ".mp3", ".flac", ".mp4", ".m4a", ".webm", ".ogg"]


class WaitState(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class CustomWaiter:
    """
    Base class for a custom waiter that leverages botocore's waiter code. Waiters
    poll an operation, with a specified delay between each polling attempt, until
    either an accepted result is returned or the number of maximum attempts is reached.

    To use, implement a subclass that passes the specific operation, arguments,
    and acceptors to the superclass.

    For example, to implement a custom waiter for the transcription client that
    waits for both success and failure outcomes of the get_transcription_job function,
    create a class like the following:

        class TranscribeCompleteWaiter(CustomWaiter):
        def __init__(self, client):
            super().__init__(
                'TranscribeComplete', 'GetTranscriptionJob',
                'TranscriptionJob.TranscriptionJobStatus',
                {'COMPLETED': WaitState.SUCCESS, 'FAILED': WaitState.FAILURE},
                client)

        def wait(self, job_name):
            self._wait(TranscriptionJobName=job_name)

    """

    def __init__(self, name, operation, argument, acceptors, client, delay=10, max_tries=60, matcher="path"):
        """
        Subclasses should pass specific operations, arguments, and acceptors to
        their superclass.

        :param name: The name of the waiter. This can be any descriptive string.
        :param operation: The operation to wait for. This must match the casing of
                          the underlying operation model, which is typically in
                          CamelCase.
        :param argument: The dict keys used to access the result of the operation, in
                         dot notation. For example, 'Job.Status' will access
                         result['Job']['Status'].
        :param acceptors: The list of acceptors that indicate the wait is over. These
                          can indicate either success or failure. The acceptor values
                          are compared to the result of the operation after the
                          argument keys are applied.
        :param client: The Boto3 client.
        :param delay: The number of seconds to wait between each call to the operation.
        :param max_tries: The maximum number of tries before exiting.
        :param matcher: The kind of matcher to use.
        """
        self.name = name
        self.operation = operation
        self.argument = argument
        self.client = client
        self.waiter_model = botocore.waiter.WaiterModel(
            {
                "version": 2,
                "waiters": {
                    name: {
                        "delay": delay,
                        "operation": operation,
                        "maxAttempts": max_tries,
                        "acceptors": [
                            {
                                "state": state.value,
                                "matcher": matcher,
                                "argument": argument,
                                "expected": expected,
                            }
                            for expected, state in acceptors.items()
                        ],
                    }
                },
            }
        )
        self.waiter = botocore.waiter.create_waiter_with_client(self.name, self.waiter_model, self.client)

    def __call__(self, parsed, **kwargs):
        """
        Handles the after-call event by logging information about the operation and its
        result.

        :param parsed: The parsed response from polling the operation.
        :param kwargs: Not used, but expected by the caller.
        """
        status = parsed
        for key in self.argument.split("."):
            if key.endswith("[]"):
                status = status.get(key[:-2])[0]
            else:
                status = status.get(key)
        logger.info("Waiter %s called %s, got %s.", self.name, self.operation, status)

    def _wait(self, **kwargs):
        """
        Registers for the after-call event and starts the botocore wait loop.

        :param kwargs: Keyword arguments that are passed to the operation being polled.
        """
        event_name = f"after-call.{self.client.meta.service_model.service_name}"
        self.client.meta.events.register(event_name, self)
        self.waiter.wait(**kwargs)
        self.client.meta.events.unregister(event_name, self)


class TranscribeCompleteWaiter(CustomWaiter):
    """
    Waits for the transcription to complete.
    """

    def __init__(self, client):
        super().__init__(
            "TranscribeComplete",
            "GetTranscriptionJob",
            "TranscriptionJob.TranscriptionJobStatus",
            {"COMPLETED": WaitState.SUCCESS, "FAILED": WaitState.FAILURE},
            client,
        )

    def wait(self, job_name):
        self._wait(TranscriptionJobName=job_name)

def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics.
    """
    return "".join(
        "_" if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
    )


def get_s3_parts(s3_uri):
    """Parse S3 URI into bucket and prefix."""
    s3_uri_parts = re.compile(r"^(?P<protocol>s3://)?(?P<bucket>.+?)/(?P<prefix>.*)$").match(s3_uri.strip())
    bucket = s3_uri_parts["bucket"]
    prefix = s3_uri_parts["prefix"]
    return bucket, prefix


def list_keys(s3_uri, client: boto3.client, max_keys=1000000) -> Set[str]:
    """Recursively get list of keys in S3 bucket."""
    keys = set()
    bucket, prefix = get_s3_parts(s3_uri)

    try:
        client.head_bucket(Bucket=bucket)

    except botocore.exceptions.ClientError as msg:
        logger.error(f"Unable to access {bucket} with error message: {msg}")
        raise SystemExit(1)

    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys):
        for obj in page.get("Contents", []):
            if Path(obj["Key"]).suffix not in FORMATS:
                continue
            keys.add(obj["Key"])

    return list(keys)


def get_delay_seconds(mean_seconds: int = 5, min_seconds: int = 1, max_seconds: int = 15):
    return min(min_seconds + random.expovariate(1 / mean_seconds), max_seconds)


@dataclass
class Transcribe:
    client: Optional[boto3.client] = field(default=None, repr=False)

    def __post_init__(self):
        self.client = boto3.client("transcribe", region_name="us-west-2")

    def get_gamma_client(self, role_arn: str):
        logging.info("Using gamma for pre-prod testing, make sure you pasted appropriate credentials")
        credentials = boto3.client("sts").assume_role(RoleArn=role_arn, RoleSessionName="test")["Credentials"]

        self.client = boto3.client(
            "transcribe",
            endpoint_url="https://gamma.transcribe.us-west-2.amazonaws.com",
            region_name="us-west-2",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )

    def start_job(
        self,
        job_name: str,
        media_uri: str,
        media_format: str,
        language_code: Optional[str] = None,
        vocabulary_name: Optional[str] = None,
        show_n_alternatives: Optional[int] = None,
    ):
        try:
            job_args = {
                "TranscriptionJobName": job_name,
                "Media": {"MediaFileUri": media_uri},
                "MediaFormat": media_format,
                "LanguageCode": "en-US",
                "Subtitles": {
                "Formats": [
                    "srt",
                ],
                "OutputStartIndex": 1,
            },
                "Settings": {
                    "ShowSpeakerLabels": True,
                    "MaxSpeakerLabels": 10,
                },
            }
            if language_code is not None:
                job_args["LanguageCode"] = language_code
            else:
                job_args["IdentifyLanguage"] = True
            if vocabulary_name is not None:
                job_args["Settings"]["VocabularyName"] = vocabulary_name

            if show_n_alternatives:
                job_args["Settings"]["ShowAlternatives"] = True
                job_args["Settings"]["MaxAlternatives"] = show_n_alternatives
            time.sleep(get_delay_seconds())
            response = self.client.start_transcription_job(**job_args)
            job = response["TranscriptionJob"]
        except ClientError as msg:
            logger.exception(f"Couldn't start transcription job {job_name}.\nError:{msg}")
        else:
            return job

    def get_job(self, job_name: str):
        try:
            time.sleep(get_delay_seconds())
            response = self.client.get_transcription_job(TranscriptionJobName=job_name)
            job = response["TranscriptionJob"]
            logger.debug(f"Got job {job['TranscriptionJobName']}")
            return job
        except ClientError as msg:
            logger.exception(f"Couldn't get job {job_name}.\nError: {msg}")

    def delete_job(self, job_name: str):
        try:
            time.sleep(get_delay_seconds())
            self.client.delete_transcription_job(TranscriptionJobName=job_name)
            logger.debug(f"Deleted job {job_name}.")
        except ClientError as msg:
            logger.exception(f"Couldn't delete job {job_name}.\nError: {msg}")
            return


@dataclass
class Vocab:
    cv_s3_path: str
    vocab_name: str
    locale: str = field(default="en-US", init=False)
    client: Optional[boto3.client] = field(default=None, repr=False)

    def __post_init__(self):
        self.client = boto3.client("transcribe", region_name="us-west-2")

        # create CV
        logging.info(
            f"Creating custom vocabulary name {self.vocab_name} based on CV list provided in {self.cv_s3_path}"
        )

        self.client.create_vocabulary(
            LanguageCode=self.locale, VocabularyName=self.vocab_name, VocabularyFileUri=self.cv_s3_path
        )

        timeout = time.time() + 600  # 10 minutes from now
        while True:
            status = self.client.get_vocabulary(VocabularyName=self.vocab_name)
            if status["VocabularyState"] in ["READY", "FAILED"] or time.time() > timeout:
                break
            logging.info("CV not ready yet...")
            time.sleep(5)
        logging.info(f"CV status is {status}")


def transcribe_worker(
    audio_file_or_s3_key: str,
    s3_bucket_name: str,
    output_dir: str,
    region="us-west-2",
    gamma_arn: Optional[str] = None,
    vocab_name: Optional[str] = None,
    show_n_alternatives: Optional[int] = None,
):
    """Worker function that transcribes single input file."""
    time.sleep(get_delay_seconds())
    transcribe = Transcribe(client=boto3.client("transcribe", region_name=region))
    if gamma_arn:
        transcribe.get_gamma_client(gamma_arn)

    recording_id = Path(audio_file_or_s3_key).stem
    ext = Path(audio_file_or_s3_key).suffix
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    job_name_simple = remove_symbols(unidecode(f"{timestamp}_{recording_id}"))
    output_filename = Path(audio_file_or_s3_key).with_suffix(".json")
    if audio_file_or_s3_key.startswith("s3://"):
        output_filename = Path(get_s3_parts(audio_file_or_s3_key)[1]).with_suffix(".json")
    media_uri = (
        audio_file_or_s3_key
        if audio_file_or_s3_key.startswith("s3://")
        else f"s3://{s3_bucket_name}/{audio_file_or_s3_key}"
    )

    if Path(audio_file_or_s3_key).is_file():
        boto3.client("s3", region_name=region).upload_file(
            audio_file_or_s3_key, s3_bucket_name, audio_file_or_s3_key
        )

    transcribe.start_job(
        job_name=job_name_simple,
        media_uri=media_uri,
        media_format=ext[1:],
        vocabulary_name=vocab_name,
        show_n_alternatives=show_n_alternatives,
    )
    logging.info(f"Started transcription job {job_name_simple} for {media_uri}.")

    transcribe_waiter = TranscribeCompleteWaiter(transcribe.client)
    time.sleep(get_delay_seconds())
    transcribe_waiter.wait(job_name_simple)

    job_simple = transcribe.get_job(job_name_simple)
    if job_simple is None:
        logging.warning(f"Error getting job {job_name_simple}.")
        return {}

    if job_simple["TranscriptionJobStatus"] != "COMPLETED":
        logging.warning(f"Job {job_name_simple} did not complete successfully.\nJob info:\n{job_simple}")
        out = {"file": job_simple["Media"]["MediaFileUri"], "failure_reason": job_simple.get("FailureReason")}
        transcribe.delete_job(job_name_simple)
        (Path(output_dir) / "failures" / f"{job_name_simple}.json").write_text(json.dumps(out))
        return out

    s3_client = boto3.client("s3", region_name="us-west-2")
    output_fn = Path(output_dir) / "json" / output_filename
    output_fn.parent.mkdir(parents=True, exist_ok=True)

    with output_fn.open("r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    transcribe.delete_job(job_name_simple)
    time.sleep(get_delay_seconds())


def main(args):
    start = time.perf_counter()

    timestr = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir = Path(args.output_dir) / timestr if not args.skip_timestamp else Path(args.output_dir)

    for path in ["json", "ctm", "trn", "failures"]:
        (output_dir / path).mkdir(parents=True, exist_ok=True)
    s3_bucket_name, _ = get_s3_parts(args.s3_uri)

    audio_source = args.manifest or args.data_dir

    if audio_source is None:
        s3_client = boto3.client("s3", region_name=args.region)
        audio_keys = list_keys(args.s3_uri, s3_client)
        audio_source = args.s3_uri
    elif Path(audio_source).is_file():
        audio_keys = Path(audio_source).read_text().splitlines()
    elif Path(audio_source).is_dir():
        audio_keys = [file for file in Path(audio_source).rglob("*") if file.suffix in FORMATS]
    else:
        raise ValueError(
            f"Audio source must be a local folder, S3 URI, or manifest file of S3 uris: {audio_source}"
        )

    assert len(audio_keys), f"No files found in {audio_source}"

    logger.info(f"Found {len(audio_keys)} files in {audio_source}. First one is: {audio_keys[0]}")

    if args.max_audio_files is not None:
        audio_keys = audio_keys[: min(args.max_audio_files, len(audio_keys))]
        logger.info(f"Running inference on subset of {len(audio_keys)} files.")

    if args.save_manifest:
        (Path(output_dir) / "audio_manifest.txt").write_text("\n".join(audio_keys))

    output_s3_uri = args.output_s3_uri.rstrip("/")
    if not args.skip_timestamp:
        output_s3_uri = f"{output_s3_uri}/{timestr}"

    max_workers = min(min(args.max_workers, len(audio_keys)), 200)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                transcribe_worker,
                audio_file_or_s3_key=key,
                s3_bucket_name=s3_bucket_name,
                output_dir=output_dir,
                output_s3_uri=output_s3_uri,
                vocab_name=args.vocab_name,
                show_n_alternatives=args.show_n_alternatives,
                region=args.region,
                gamma_arn=args.gamma_role_arn,
            )
            for key in audio_keys
        ]
    wait(futures)

    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    failures = [res for res in results if res]

    logging.info(
        f"Done. Completed  transcription for data in {audio_source}.\n"
        f"Processed N={len(audio_keys)}. Failures: N={len(failures)} of {len(audio_keys)}.\n"
        f"Elapsed time: {(time.perf_counter() - start)/60: .4f} mins."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon Transcribe  transcription inference.")
    parser.add_argument("s3_uri", help="S3 URI containing audio files.")
    data_source_group = parser.add_mutually_exclusive_group()
    data_source_group.add_argument("--manifest", help="Manifest file of S3 URIs.")
    data_source_group.add_argument("--data_dir", help="Local data directory")
    parser.add_argument("--output_dir", help="Output directory to write results", required=True)
    parser.add_argument(
        "--output_s3_uri",
        help="Output S3 path to write results",
        required=True,
    )
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--skip_timestamp", action="store_true", help="Skip timestamping output directory.")
    parser.add_argument(
        "--save_manifest", action="store_true", help="Save audio manifest to output directory."
    )
    parser.add_argument("--show_n_alternatives", type=int, help="Show alternative transcripts")
    vocab_group = parser.add_mutually_exclusive_group()
    vocab_group.add_argument("--cv_s3_path", help="Input S3 path to CV list")
    vocab_group.add_argument("--vocab_name", help="Custom vocabulary name")
    parser.add_argument("--gamma_role_arn", help="Gamma role ARN. For use with testing with Gamma account.")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=200,
        help="Maximum number of concurrent Transcribe jobs. Check account quota https://docs.aws.amazon.com/general/latest/gr/transcribe.html.",
    )
    parser.add_argument(
        "--max_audio_files",
        type=int,
        help="Limits total audios processed (0=no limit)",
    )
    parser.add_argument(
        "--log_level",
        default="info",
        type=str,
        choices=["debug", "info", "warning", "error"],
    )
    return parser.parse_args()



def cli():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=args.log_level.upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main(args)


if __name__ == "__main__":
    cli()
