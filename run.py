import os
import uvicorn

if __name__ == "__main__":
    # 📑 Dynamic port assignment injected by the cloud platform
    # Defaults to 8000 for local environment execution fallback
    port = int(os.getenv("PORT", 8000))
    
    print(f"[INFO] Launching production network server on host 0.0.0.0 listening on port: {port}")
    uvicorn.run(
        "backend.app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False
    )