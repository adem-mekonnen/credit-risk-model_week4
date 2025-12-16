from pydantic import BaseModel

class CreditScoringRequest(BaseModel):
    TotalAmount: float
    AvgAmount: float
    StdAmount: float
    TxCount: int
    TotalValue: float
    AvgValue: float
    ProductCategory: int
    ChannelId: int
    PricingStrategy: int

class CreditScoringResponse(BaseModel):
    risk_probability: float
    is_high_risk: int