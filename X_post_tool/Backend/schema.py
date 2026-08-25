from pydantic import BaseModel, Field
from typing import List, Optional


class LearningPoint(BaseModel):
    """Represents an individual learning section/point with heading and bullet items."""
    point_number: int = Field(
        ..., 
        description="Sequential number of the learning point (e.g., 1, 2, 3)."
    )
    heading: str = Field(
        ..., 
        description="Clear and concise heading for this specific learning point."
    )
    points: List[str] = Field(
        default_factory=list, 
        description="List of detailed bullet points, facts, or explanations for this heading."
    )
    explanation: Optional[str] = Field(
        default=None, 
        description="Optional in-depth context or elaboration for this point."
    )


class LearningResponse(BaseModel):
    """
    Structured response model to convert unstructured paragraphs 
    into well-organized learning points with headings, bullets, and key takeaways.
    """
    initial_heading: str = Field(
        ..., 
        description="The primary main heading or topic title summarizing the response."
    )
    overview: Optional[str] = Field(
        default=None, 
        description="A brief 1-2 sentence high-level overview or introduction."
    )
    learning_points: List[LearningPoint] = Field(
        default_factory=list, 
        description="List of core learning points with subheadings and bullet points."
    )
    key_takeaways: List[str] = Field(
        default_factory=list, 
        description="List of core key takeaways or summary highlights."
    )

    def to_markdown(self) -> str:
        """
        Converts the structured learning response into formatted markdown.
        """
        lines = [f"# {self.initial_heading}\n"]

        if self.overview:
            lines.append(f"{self.overview}\n")

        if self.learning_points:
            lines.append("## Learning Points\n")
            for lp in self.learning_points:
                lines.append(f"### {lp.point_number}. {lp.heading}")
                for pt in lp.points:
                    lines.append(f"- {pt}")
                if lp.explanation:
                    lines.append(f"\n> {lp.explanation}\n")
                lines.append("")

        if self.key_takeaways:
            lines.append("## 📌 Key Takeaways\n")
            for takeaway in self.key_takeaways:
                lines.append(f"- {takeaway}")
            lines.append("")

        return "\n".join(lines).strip()