import operator
from typing import Annotated, List, Literal, Sequence

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State toàn cục LangGraph — reducer messages cộng dồn."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    raw_data: str
    chart_path: str
    revision_count: int
    sender: str
    status: str


class ResearcherOutput(BaseModel):
    summary: str = Field(
        description="Tóm tắt khách quan các số liệu / sự kiện từ nguồn đã tìm."
    )
    sources: List[str] = Field(
        default_factory=list,
        description="URL hoặc tên nguồn đã dùng.",
    )


class CriticDecision(BaseModel):
    decision: Literal["APPROVE", "REJECT"] = Field(
        description="APPROVE nếu đạt; REJECT nếu cần làm lại."
    )
    feedback: str = Field(
        description="Nếu REJECT: lý do cụ thể. Nếu APPROVE: một câu ngắn.",
    )
