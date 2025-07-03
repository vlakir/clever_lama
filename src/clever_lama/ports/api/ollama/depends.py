from typing import Annotated

from fastapi import Depends

from clever_lama.services.proxy import OpenAIService

OpenAIServiceDep = Annotated[OpenAIService, Depends(OpenAIService)]
