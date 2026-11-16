"""Fail-closed API key authentication; bind to loopback for demonstrations."""
import os
import secrets
from functools import lru_cache
from typing import Annotated
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from .service import Engine, Conflict

app=FastAPI(title='FinGuard',version='0.1.0',description='Synthetic financial-network risk research; scores are not findings of criminal activity.')
class Transaction(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    tx_id: str=Field(min_length=1,max_length=80,pattern=r'^\S+$')
    step: int=Field(ge=0,strict=True)
    sender: str=Field(min_length=1,max_length=80,pattern=r'^\S+$')
    receiver: str=Field(min_length=1,max_length=80,pattern=r'^\S+$')
    amount: float=Field(ge=0,le=1e12)
class Batch(BaseModel):
    model_config=ConfigDict(extra='forbid')
    transactions: list[Transaction]=Field(min_length=1,max_length=1000)

def auth(role):
    def check(x_api_key: Annotated[str|None,Header()]=None):
        expected=os.getenv('FINGUARD_'+role+'_KEY','')
        if len(expected)<24: raise HTTPException(503,'Configure a key with at least 24 characters')
        if not x_api_key or not secrets.compare_digest(x_api_key,expected): raise HTTPException(401,'Invalid API key')
    return check

@lru_cache
def engine():
    return Engine(os.getenv('FINGUARD_DB','finguard.db'),os.getenv('FINGUARD_MODELS','models'))

@app.get('/health')
def health(): return {'status':'alive','version':'0.1.0'}

@app.post('/transactions',dependencies=[Depends(auth('INGEST'))])
def ingest(batch:Batch):
    try: return {'transactions':engine().ingest([t.model_dump() for t in batch.transactions])}
    except Conflict as e: raise HTTPException(409,str(e)) from e
    except ValueError as e: raise HTTPException(422,str(e)) from e
    except FileNotFoundError as e: raise HTTPException(503,'Run the experiment to create model artifacts') from e

@app.get('/alerts',dependencies=[Depends(auth('READ'))])
def alerts(limit:int=Query(100,ge=1,le=1000)):
    return {'alerts':engine().alerts(limit)}
