import os
import csv
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import RAG and Agent functionality from your codebase
from rag import setup_pipeline
from agent import run_agent

load_dotenv()

# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================

DATASET_CSV_PATH = "GoldenDataset.csv"
PDF_DATASET_DIR = "Dataset"
PDF_FILES = [
    "Paper1.pdf",
    "Paper2.pdf",
    "Paper3.pdf"
]

# Model used for LLM-as-a-Judge (OpenAI's flagship model)
JUDGE_MODEL_NAME = "gpt-4o"

# =========================================================
# LLM JUDGE STRUCTURED OUTPUT SCHEMA
# =========================================================

class EvaluationResult(BaseModel):
    """Schema for LLM Judge Evaluation Output."""
    score: float = Field(
        ..., 
        description="Accuracy/Correctness score from 0.0 (completely wrong) to 1.0 (perfect answer)."
    )
    is_correct: bool = Field(
        ..., 
        description="True if the predicted answer accurately conveys the core information in the ground truth."
    )
    reasoning: str = Field(
        ..., 
        description="Detailed explanation of why the score was assigned, highlighting matches or missing facts."
    )


# =========================================================
# LLM JUDGE CLASS
# =========================================================

class RAGJudge:
    """LLM-as-a-Judge evaluator using GPT-4o."""

    def __init__(self, model_name: str = JUDGE_MODEL_NAME):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0
        ).with_structured_output(EvaluationResult)

        self.system_prompt = SystemMessage(
            content="""You are an expert impartial evaluation judge assessing the performance of a Retrieval-Augmented Generation (RAG) system.

Your task is to compare the SYSTEM PREDICTION against the GROUND TRUTH answer for a given user question.

Evaluation Guidelines:
1. Focus on semantic accuracy, key facts, and technical precision.
2. Do NOT penalize variations in formatting, style, length, or structural markdown (bullets vs paragraphs).
3. Ignore slight variations in wording if the essential meaning and numeric values align.
4. If the predicted answer contains hallucinated or contradictory facts not supported by the ground truth, lower the score.
5. Provide a score between 0.0 and 1.0, where:
   - 1.0: Perfectly accurate and complete.
   - 0.7 - 0.9: Mostly accurate with minor non-critical omitted details.
   - 0.4 - 0.6: Partially accurate, missing key elements or partially incorrect.
   - 0.0 - 0.3: Largely incorrect, completely missing the answer, or severe hallucinations.
"""
        )

    def evaluate_pair(self, question: str, ground_truth: str, prediction: str) -> EvaluationResult:
        """Evaluates a single (question, ground_truth, prediction) triplet."""
        user_message = HumanMessage(
            content=f"""
USER QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

SYSTEM PREDICTION:
{prediction}
"""
        )
        try:
            result = self.llm.invoke([self.system_prompt, user_message])
            if isinstance(result, EvaluationResult):
                return result
            elif isinstance(result, dict):
                return EvaluationResult(**result)
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            return EvaluationResult(
                score=0.0,
                is_correct=False,
                reasoning=f"Evaluation failed due to exception: {str(e)}"
            )


# =========================================================
# MAIN RAG EVALUATOR ENGINE
# =========================================================

class RAGEvaluator:
    """Orchestrates loading data, running the RAG pipeline, and running evaluations."""

    def __init__(
        self, 
        golden_dataset_path: str = DATASET_CSV_PATH, 
        pdf_dir: str = PDF_DATASET_DIR, 
        pdf_files: List[str] = PDF_FILES
    ):
        self.dataset_path = golden_dataset_path
        self.pdf_paths = [os.path.join(pdf_dir, f) for f in pdf_files]
        self.judge = RAGJudge()
        self.doc_collection = None

    def initialize_pipeline(self):
        """Builds vector stores and retrievers for all documents."""
        print("Initializing Document Collection & Pipelines...")
        for path in self.pdf_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required dataset PDF not found at: {path}")

        file_tuples = [(p, os.path.basename(p)) for p in self.pdf_paths]
        self.doc_collection = setup_pipeline(file_tuples)
        print(f"Successfully loaded and indexed {len(file_tuples)} documents.\n")

    def run_evaluation(self, output_csv_path: str = "evaluation_results.csv") -> Dict[str, Any]:
        """Runs evaluation over all questions in GoldenDataset.csv."""
        if not self.doc_collection:
            self.initialize_pipeline()

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Golden dataset CSV not found at: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        
        # Verify columns exist
        required_cols = ["id", "Question", "Ground_truth"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Golden dataset CSV must contain column '{col}'")

        results = []
        total_score = 0.0
        correct_count = 0
        total_questions = len(df)

        print(f"Starting evaluation of {total_questions} questions...\n" + "="*60)

        for idx, row in df.iterrows():
            q_id = row["id"]
            question = str(row["Question"])
            ground_truth = str(row["Ground_truth"])

            print(f"[{idx + 1}/{total_questions}] Evaluating QID {q_id}: {question[:60]}...")

            # 1. Run RAG Pipeline Agent
            prediction = run_agent(
                question=question,
                history=[],
                doc_collection=self.doc_collection
            )

            # 2. Judge Prediction
            eval_res = self.judge.evaluate_pair(question, ground_truth, prediction)

            total_score += eval_res.score
            if eval_res.is_correct:
                correct_count += 1

            # 3. Save evaluation row
            results.append({
                "id": q_id,
                "Question": question,
                "Ground_truth": ground_truth,
                "Prediction": prediction,
                "Score": eval_res.score,
                "Is_Correct": eval_res.is_correct,
                "Reasoning": eval_res.reasoning
            })

            print(f" -> Score: {eval_res.score:.2f} | Correct: {eval_res.is_correct}")

        # Summary Statistics
        overall_accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0.0
        mean_score = (total_score / total_questions) if total_questions > 0 else 0.0

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions Evaluated: {total_questions}")
        print(f"Strict Binary Accuracy:   {overall_accuracy:.2f}% ({correct_count}/{total_questions})")
        print(f"Mean Accuracy Score:     {mean_score * 100:.2f}%")
        print("="*60)

        # Save to detailed CSV output
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv_path, index=False)
        print(f"\nDetailed evaluation report saved to '{output_csv_path}'.")

        return {
            "total_questions": total_questions,
            "overall_accuracy_pct": overall_accuracy,
            "mean_score": mean_score,
            "results": results
        }


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()