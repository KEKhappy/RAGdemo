import traceback

from fastapi import APIRouter, Depends, HTTPException
from pydantic_core import ValidationError
from app.schemas.schemas import QuestionRequest, AnswerResponse, ErrorResponse
from app.core.rag_service import RAGService
from app.dependencies import get_rag_service

router = APIRouter(tags=["RAG_Question_Answering"])

@router.post(
    "/query",
    response_model=AnswerResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def ask_question(
        request: QuestionRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    try:
        result = rag_service.query(question=request.question)
        print("Got result, returning...")
        return AnswerResponse(answer=result)

    except ValidationError as e:
        error_details = "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
            for err in e.errors()
        )
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error="Validation Error",
                details=error_details
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Value Error", "details": str(e)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Error", "details": str(e)}
        )