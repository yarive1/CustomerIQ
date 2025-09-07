# CustomerIQ Backend

FastAPI backend for customer segmentation and campaign management.

## Environment Setup

1. **Create Environment File**
   ```bash
   cp .env.example .env
   ```

2. **Configure Environment Variables**
   Edit `.env` file with your actual values:
   ```
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_service_role_key
   N8N_WEBHOOK_URL=http://localhost:5678/webhook-test/campaign-trigger
   ```

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or install individually:
   ```bash
   pip install fastapi uvicorn pandas numpy joblib supabase python-dotenv httpx
   ```

2. **Start Server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `POST /segment-and-store` - Upload CSV and segment customers
- `POST /campaigns/start` - Start new campaign
- `GET /campaigns/recent` - Get recent campaigns
- `POST /campaign-success` - n8n webhook for campaign completion
- `POST /trigger-n8n-campaign` - Trigger n8n workflow
- `GET /segments/customer-counts` - Get customer segment counts
- `GET /health` - Health check

## Security

- All sensitive keys are stored in environment variables
- Supabase configuration loaded from `.env`
- CORS configured for frontend integration

## File Structure

```
segmentationbackend/
├── main.py                 # Main FastAPI application
├── .env                    # Environment variables (not committed)
├── .env.example           # Environment template
├── .gitignore             # Git ignore rules
├── customer_segmentation_model.joblib  # ML model
└── README.md             # This file
```
