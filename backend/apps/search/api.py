from ninja import Router
from ninja.errors import HttpError

from core.auth import jwt_auth
from core.permissions import check_plan_limit
from .schemas import SearchIn, SearchOut, SourceOut, ConfidenceOut, ConfidenceBreakdownOut
from .services import search_service

router = Router()


@router.post(
    "/query",
    auth=jwt_auth,
    response=SearchOut,
    summary="جستجوی هوشمند در قوانین",
)
def query(request, data: SearchIn):
    user = request.user

    # چک plan limit و increment در یک atomic operation
    if not user.can_query_and_increment():
        raise HttpError(
            429,
            "سقف روزانه پرسش‌های شما تمام شده. برای ادامه پلن خود را ارتقا دهید."
        )

    if len(data.query.strip()) < 5:
        raise HttpError(400, "سوال خیلی کوتاه است.")

    result = search_service.search(data.query, data.law)

    if not result:
        raise HttpError(500, "خطا در پردازش درخواست. لطفاً دوباره تلاش کنید.")

    confidence = result["confidence"]

    return SearchOut(
        answer=result["answer"],
        confidence=ConfidenceOut(
            final=confidence.final,
            level=confidence.level,
            note=confidence.note,
            breakdown=ConfidenceBreakdownOut(
                embedding=confidence.embedding,
                llm=confidence.llm,
                graph=confidence.graph,
            ),
        ),
        sources=[
            SourceOut(
                article_number=a.num,
                law=a.law,
                source_type="law",
            )
            for a in result["sources"]
        ],
    )



@router.get(
    "/laws",
    auth=jwt_auth,
    summary="لیست قوانین موجود در سیستم",
)
def available_laws(request):
    """
    لیست قوانینی که در Neo4j موجودند.
    با اضافه شدن قوانین جدید، خودکار آپدیت میشه.
    """
    with search_service.driver.session() as session:
        result = session.run(
            "MATCH (l:Law) RETURN l.name AS name ORDER BY l.name"
        )
        return {"laws": [r["name"] for r in result]}