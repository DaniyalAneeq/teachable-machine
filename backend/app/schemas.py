from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str


class DatasetClassSummary(BaseModel):
    class_name: str
    image_count: int
    images: list[str] = []


class DatasetSummaryResponse(BaseModel):
    total_classes: int
    total_images: int
    classes: list[DatasetClassSummary]


class ModelStatusResponse(BaseModel):
    model_exists: bool
    model_path: str | None
    classes: list[str]
    feature_extractor: str | None
    image_size: int | None
