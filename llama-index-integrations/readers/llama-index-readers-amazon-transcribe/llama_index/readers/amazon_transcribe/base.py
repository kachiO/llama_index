import logging
from dataclasses import dataclass, field
from typing import Optional

import boto3  # type: ignore
from botocore.exceptions import ClientError

from enum import Enum


from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


logger = logging.getLogger(__name__)


class MediaFormats(Enum):
    AMR = "amr"
    FLAC = "flac"
    M4A = "m4a"
    MP3 = "mp3"
    MP4 = "mp4"
    OGG = "ogg"
    WAV = "wav"
    WEBM = "webm"


def try_except(error_message):
    """Decorator that wraps a function in a try/except block with a customizable message."""

    def wrapper(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ClientError as msg:
                message = error_message or ""
                logger.exception(f"{message}Transcription job error:\n{msg}")
            return inner

    return wrapper


@dataclass
class TranscribeJob:
    client: Optional[boto3.client] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.client = boto3.client("transcribe", region_name="us-west-2")

    @try_except("Start ")
    def start(
        self,
        job_name: str,
        media_uri: str,
        media_format: str,
        language_code: Optional[str] = None,
        vocabulary_name: Optional[str] = None,
        show_n_alternatives: Optional[int] = None,
    ):
        job_args = {
            "TranscriptionJobName": job_name,
            "Media": {"MediaFileUri": media_uri},
            "MediaFormat": media_format,
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
        response = self.client.start_transcription_job(**job_args)
        logger.debug(f"Started job {job['TranscriptionJobName']}")
        return response["TranscriptionJob"]

    @try_except("Get ")
    def get(self, job_name: str):
        response = self.client.get_transcription_job(TranscriptionJobName=job_name)
        job = response["TranscriptionJob"]
        logger.debug(f"Got job {job['TranscriptionJobName']}")
        return job

    @try_except("Delete ")
    def delete(self, job_name: str):
        self.client.delete_transcription_job(TranscriptionJobName=job_name)
        logger.debug(f"Deleted job {job_name}.")


class AmazonTranscribeReader(BaseReader):
    """Reader for Amazon Transcribe transcripts."""

    def __init__(self, s3_uri_or_file_path: str):
        self.file = s3_uri_or_file_path
        self.transcribe = Transcribe(
            client=boto3.client("transcribe", region_name=region)
        )

    def load_data(self) -> List[Document]:
        recording_id = Path(self.file).stem
        ext = Path(self.file).suffix
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        job_name_simple = remove_symbols(unidecode(f"{timestamp}_{recording_id}"))
        media_uri = (
            self.file
            if self.file.startswith("s3://")
            else f"s3://{s3_bucket_name}/{self.file}"
        )

        if Path(self.file).is_file():
            boto3.client("s3", region_name=region).upload_file(
                self.file, s3_bucket_name, self.file
            )

        raise NotImplementedError
