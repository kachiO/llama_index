from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class AmazonTranscribeReader(BaseReader):
    """Reader for Amazon Transcribe transcripts."""

    def __init__(self, s3_uri_or_file_path: str):
        self.file = s3_uri_or_file_path
        self.client = boto3.client("transcribe")

    def load_data(self) -> List[Document]:
        raise NotImplementedError
