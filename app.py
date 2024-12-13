from fastapi import FastAPI, UploadFile, File, HTTPException
from io import StringIO
import pandas as pd
import joblib
import uvicorn
from fastapi.responses import JSONResponse
from Binary_smokers_V_10_12 import preprocess_data


MODEL_FILE = "voting_model3.pkl"

try:
    voting_model3 = joblib.load(MODEL_FILE)

except FileNotFoundError as e:
    raise RuntimeError(f"Ошибка загрузки моделей: {e}")


app = FastAPI()


@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    """
    Принимает CSV файл с данными, выполняет предобработку и возвращает предсказания.
    """
    try:

        content = await file.read()
        data = pd.read_csv(StringIO(content.decode("utf-8")), index_col=0)


        data_preprocessed = preprocess_data(data)


        predictions = voting_model3.predict(data_preprocessed)
        class_names = ['Class 0', 'Class 1']


        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "input_row": data.iloc[i].to_dict(),
                "predicted_class": class_names[pred],
            })


        return JSONResponse(content={"predictions": results})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


