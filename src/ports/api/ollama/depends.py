from typing import Annotated

from fastapi import Depends

from services.proxy import OpenAIService

OpenAIServiceDep = Annotated[OpenAIService, Depends(OpenAIService)]
