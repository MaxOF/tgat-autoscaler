from fastapi import APIRouter, HTTPException

from app.service.tgat_service import TGATService
from app.dto.train_csv_request_dto import TrainCSVRequest
from config.settings import NODES_CSV_PATH, EDGES_CSV_PATH

router = APIRouter(prefix='/api')
tgat_service = TGATService()


@router.post('/train')
async def train():
    try:        
        training_request = TrainCSVRequest(nodes_csv_path=NODES_CSV_PATH, edges_csv_path=EDGES_CSV_PATH)
        print(training_request)
        response = await tgat_service.train_from_csv(training_request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post('/ablate')
def ablate():
    try:        
        response = tgat_service.ablate()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ablate failed: {str(e)}")

@router.post('/predict')
async def predict():
    try:        
        response = await tgat_service.predict()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ablate failed: {str(e)}")

@router.post('/apply')
async def apply():
    try:        
        response = await tgat_service.apply()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ablate failed: {str(e)}")