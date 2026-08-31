import os
from dotenv import load_dotenv

load_dotenv()

# Forcefully disable LangSmith tracing during evaluation if API keys are missing/invalid
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TEST_TRACKING"] = "false"

import csv
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import RAG retrieval setup
from rag import setup_pipeline, DocumentCollection

# Initialize evaluation LLM and Judge LLM
eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
JUDGE_MODEL_NAME = "gpt-4o"

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

# =========================================================
# EVALUATION-SPECIFIC CONCISE RAG EXECUTION NODE
# =========================================================

def execute_eval_rag(
    question: str,
    doc_collection: DocumentCollection,
    mode: str = "hybrid"
) -> str:
    """
    Executes RAG retrieval but returns a single concise paragraph 
    without UI learning points or markdown headers.
    """
    num_docs = doc_collection.document_count
    top_k_per_doc = 3 if num_docs > 1 else 6

    context = doc_collection.get_formatted_context(
        question=question,
        mode=mode,
        top_k_per_doc=top_k_per_doc
    )

    system_prompt = SystemMessage(
        content="""You are a precise technical Q&A assistant analyzing research papers.
Answer the question accurately using ONLY the provided document context.

CONSTRAINTS FOR EVALUATION:
- Respond in ONE single, concise, and direct paragraph.
- Do NOT use markdown headers, bullet points, numbered lists, or "Learning Points" sections.
- Keep the tone direct and strictly factual to match the ground truth.

DOCUMENT CONTEXT:
""" + context
    )

    messages = [system_prompt, HumanMessage(content=question)]
    response = eval_llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


# =========================================================
# LLM JUDGE STRUCTURED OUTPUT SCHEMA
# =========================================================

class EvaluationResult(BaseModel):
    """Schema for LLM Judge Evaluation Output."""
    score: float = Field(
        ..., 
        description="Accuracy score from 0.0 (completely incorrect) to 1.0 (perfectly accurate)."
    )
    is_correct: bool = Field(
        ..., 
        description="True if the prediction accurately conveys the core information in the ground truth."
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation of why the score was assigned."
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
2. Ignore minor stylistic variations if the core factual meaning matches.
3. If the predicted answer contains hallucinated facts or misses essential factual content from ground truth, penalize the score.
4. Assign a score between 0.0 and 1.0:
   - 1.0: Fully accurate and complete.
   - 0.7 - 0.9: Mostly accurate with minor omitted details.
   - 0.4 - 0.6: Partially accurate or incomplete.
   - 0.0 - 0.3: Largely incorrect or hallucinatory.
"""
        )

    def evaluate_pair(self, question: str, ground_truth: str, prediction: str) -> EvaluationResult:
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
# MAIN COMPARATIVE RAG EVALUATOR ENGINE
# =========================================================

class RAGComparativeEvaluator:
    """Evaluates and compares Standard RAG vs Hybrid RAG performance."""

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
        print("Initializing Document Collection & Vector/BM25 Indexes...")
        for path in self.pdf_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required dataset PDF not found at: {path}")

        file_tuples = [(p, os.path.basename(p)) for p in self.pdf_paths]
        self.doc_collection = setup_pipeline(file_tuples)
        print(f"Successfully loaded and indexed {len(file_tuples)} documents.\n")

    def run_evaluation(self, output_csv_path: str = "rag_comparison_results.csv") -> Dict[str, Any]:
        if not self.doc_collection:
            self.initialize_pipeline()

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Golden dataset CSV not found at: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        df.columns = df.columns.str.strip()

        required_cols = ["id", "Question", "Ground_truth"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Golden dataset CSV must contain column '{col}'. Found: {list(df.columns)}")

        results = []
        std_total_score = 0.0
        std_correct_count = 0

        hybrid_total_score = 0.0
        hybrid_correct_count = 0

        total_questions = len(df)

        print(f"Starting Comparative Evaluation over {total_questions} questions...\n" + "="*80)

        for idx, row in df.iterrows():
            q_id = row["id"]
            question = str(row["Question"])
            ground_truth = str(row["Ground_truth"])

            print(f"\n[{idx + 1}/{total_questions}] Question ID {q_id}: {question[:65]}...")

            # 1. Standard RAG (Dense Only - Concise Paragraph)
            std_prediction = execute_eval_rag(
                question=question,
                doc_collection=self.doc_collection,
                mode="standard"
            )
            std_eval = self.judge.evaluate_pair(question, ground_truth, std_prediction)

            std_total_score += std_eval.score
            if std_eval.is_correct:
                std_correct_count += 1

            # 2. Hybrid RAG (Dense + BM25 + RRF - Concise Paragraph)
            hybrid_prediction = execute_eval_rag(
                question=question,
                doc_collection=self.doc_collection,
                mode="hybrid"
            )
            hybrid_eval = self.judge.evaluate_pair(question, ground_truth, hybrid_prediction)

            hybrid_total_score += hybrid_eval.score
            if hybrid_eval.is_correct:
                hybrid_correct_count += 1

            print(f"  ├─ [Standard RAG] Score: {std_eval.score:.2f} | Correct: {std_eval.is_correct}")
            print(f"  └─ [Hybrid RAG]   Score: {hybrid_eval.score:.2f} | Correct: {hybrid_eval.is_correct}")

            results.append({
                "id": q_id,
                "Question": question,
                "Ground_truth": ground_truth,
                "Standard_Prediction": std_prediction,
                "Standard_Score": std_eval.score,
                "Standard_Is_Correct": std_eval.is_correct,
                "Standard_Reasoning": std_eval.reasoning,
                "Hybrid_Prediction": hybrid_prediction,
                "Hybrid_Score": hybrid_eval.score,
                "Hybrid_Is_Correct": hybrid_eval.is_correct,
                "Hybrid_Reasoning": hybrid_eval.reasoning
            })

        # Calculations
        std_accuracy = (std_correct_count / total_questions) * 100 if total_questions > 0 else 0.0
        std_mean_score = (std_total_score / total_questions) * 100 if total_questions > 0 else 0.0

        hybrid_accuracy = (hybrid_correct_count / total_questions) * 100 if total_questions > 0 else 0.0
        hybrid_mean_score = (hybrid_total_score / total_questions) * 100 if total_questions > 0 else 0.0

        # Output Summary
        print("\n" + "="*80)
        print("                        FINAL RAG EVALUATION COMPARISON")
        print("="*80)
        print(f"Total Evaluated Questions: {total_questions}\n")
        print(f"{'Metric':<30} | {'Standard RAG':<20} | {'Hybrid RAG':<20}")
        print("-" * 80)
        print(f"{'Strict Binary Accuracy':<30} | {f'{std_accuracy:.2f}% ({std_correct_count}/{total_questions})':<20} | {f'{hybrid_accuracy:.2f}% ({hybrid_correct_count}/{total_questions})':<20}")
        print(f"{'Mean Accuracy Score':<30} | {f'{std_mean_score:.2f}%':<20} | {f'{hybrid_mean_score:.2f}%':<20}")
        print("="*80)

        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv_path, index=False)
        print(f"\nDetailed evaluation report saved to '{output_csv_path}'.")

        return {
            "total_questions": total_questions,
            "standard_rag": {
                "binary_accuracy_pct": std_accuracy,
                "mean_accuracy_score_pct": std_mean_score
            },
            "hybrid_rag": {
                "binary_accuracy_pct": hybrid_accuracy,
                "mean_accuracy_score_pct": hybrid_mean_score
            }
        }


if __name__ == "__main__":
    evaluator = RAGComparativeEvaluator()
    evaluator.run_evaluation()