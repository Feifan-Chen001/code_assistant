from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel
import pydantic

class ReviewFinding(BaseModel):
    tool: str
    rule: str
    severity: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    extra: Dict[str, Any] = {}
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """兼容 Pydantic 2.x 的 model_dump 方法
        
        Pydantic 1.x 使用 dict()，2.x 使用 model_dump()
        这个方法让代码在两个版本上都能工作
        """
        # 直接调用 BaseModel 的 dict 方法（不通过我们的 model_dump）
        # 检查 Pydantic 版本
        if hasattr(pydantic, '__version__') and pydantic.__version__.startswith('2'):
            # Pydantic 2.x
            return BaseModel.model_dump(self, **kwargs)
        else:
            # Pydantic 1.x - 调用原始的 dict 方法
            # 注意：BaseModel.__dict__ 是类属性，我们需要调用 dict 实例方法
            # 使用 object.__getattribute__ 来绕过任何自定义的 __getattribute__
            return BaseModel.dict(self, **{k: v for k, v in kwargs.items() if k in ('by_alias', 'exclude_unset', 'exclude_defaults', 'exclude_none')})
