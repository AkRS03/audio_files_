from pydantic import BaseModel

class DownloadRequest(BaseModel):
    url: str
    username: str
    password: str
    order_id: int
    SRN_id: int

class DownloadRequestResponse(BaseModel):
    url: str
    username: str
    order_id: int
    SRN_id: int
    File_size: float
    Duration: float