import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from datetime import datetime
from pydantic import BaseModel
import warnings
import io
import re
import uuid
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Settings ---
warnings.filterwarnings('ignore')

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Customer Segmentation API",
    description="Upload a CSV, segment customers, and store them in Supabase."
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define the Mapping to Your Exact Supabase Table Names ---
TABLE_NAME_MAP = {
    '1. New & Cautious': 'new_and_cautious',
    '2. Stable Earners': 'stable_earners',
    '3. Mid-Tier Professionals': 'mid_tier_professionals',
    '4. Affluent Customers': 'affluent_customers',
    '5. High-Value Elite': 'high_value_elite'
}

# --- Load Model and Connect to Supabase ---
try:
    model_pipeline = joblib.load('customer_segmentation_model.joblib')
    print("✅ Trained model loaded successfully.")

    # Get Supabase configuration from environment variables
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing Supabase configuration in environment variables")
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase client initialized.")

except FileNotFoundError:
    print("❌ FATAL ERROR: 'customer_segmentation_model.joblib' not found.")
    model_pipeline = None
except ValueError as e:
    print(f"❌ CONFIGURATION ERROR: {e}")
    supabase = None

# --- API Endpoint ---
@app.post("/segment-and-store")
async def segment_and_store(file: UploadFile = File(...)):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        contents = await file.read()
        df_new = pd.read_csv(io.BytesIO(contents))
        
        data_to_segment = df_new.copy()
        data_to_segment.replace(['_INVALID_', '_RARE_'], np.nan, inplace=True)
        data_to_segment.dropna(inplace=True)
        
        for col in ['has_loan', 'has_credit_card', 'has_investment']:
            if col in data_to_segment.columns:
                data_to_segment[col] = data_to_segment[col].astype(int)
        for col in ['age', 'income', 'balance', 'account_tenure']:
            if col in data_to_segment.columns:
                data_to_segment[col] = pd.to_numeric(data_to_segment[col], errors='coerce')
        data_to_segment.dropna(inplace=True)

        predicted_clusters = model_pipeline.predict(data_to_segment)
        data_to_segment['cluster'] = predicted_clusters
        
        print(f"✅ Successfully segmented {len(data_to_segment)} customers.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

    # --- Data-Driven Naming Logic ---
    cluster_summary = data_to_segment.groupby('cluster')[['income', 'balance', 'age']].mean()
    cluster_summary.sort_values(by=['income', 'balance'], inplace=True)
    
    sorted_persona_names = list(TABLE_NAME_MAP.keys())
    
    final_label_map = {cluster_id: name for cluster_id, name in zip(cluster_summary.index, sorted_persona_names)}
    data_to_segment['persona_name'] = data_to_segment['cluster'].map(final_label_map)
    print("✅ Persona names assigned.")

    # --- Store Data in Supabase ---
    results = {"message": "Segmentation successful.", "clusters": {}}
    
    for persona_name in sorted_persona_names:
        table_name = TABLE_NAME_MAP.get(persona_name)
        if not table_name:
            continue

        cluster_df = data_to_segment[data_to_segment['persona_name'] == persona_name]
        
        if not cluster_df.empty:
            records = cluster_df.drop(columns=['cluster', 'persona_name']).to_dict(orient='records')
            try:
                # --- (FIX) Use .upsert() instead of .insert() ---
                # This will update existing records and insert new ones.
                data, count = supabase.table(table_name).upsert(records, on_conflict='customer_id').execute()
                
                results["clusters"][table_name] = {"processed_count": len(records), "status": "success"}
                print(f"  -> Upserted {len(records)} records into {table_name}")
            except Exception as e:
                results["clusters"][table_name] = {"processed_count": 0, "status": "failed", "error": str(e)}
                print(f"  -> FAILED to upsert into {table_name}: {e}")

    return results

# --- n8n Campaign Success Webhook ---
@app.post("/campaign-success")
async def campaign_success_webhook(request_data: dict):
    """
    Webhook endpoint to receive campaign completion notification from n8n
    Updates campaign status in database
    """
    try:
        print(f"📧 Campaign success notification received: {request_data}")
        print(f"📊 Request data keys: {list(request_data.keys())}")
        
        campaign_id = request_data.get("campaign_id")
        if not campaign_id:
            print(f"❌ Missing campaign_id in request: {request_data}")
            raise HTTPException(status_code=400, detail=f"campaign_id is required. Received: {list(request_data.keys())}")
        
        print(f"📋 Processing campaign completion for: {campaign_id}")
        
        # Calculate completion time on our backend
        completed_at = datetime.now().isoformat()
        
        # Get the campaign details to find segment and calculate customer count
        emails_sent = 0
        try:
            campaign_result = supabase.table("campaign_logs")\
                .select("segment_name")\
                .eq("campaign_id", campaign_id)\
                .execute()
            
            if campaign_result.data and len(campaign_result.data) > 0:
                segment_name = campaign_result.data[0].get("segment_name")
                table_name = TABLE_NAME_MAP.get(segment_name)
                
                if table_name:
                    try:
                        count_result = supabase.table(table_name).select("customer_id", count="exact").execute()
                        emails_sent = count_result.count if count_result.count is not None else 0
                        print(f"📧 Calculated {emails_sent} emails sent for {segment_name} segment")
                    except Exception as e:
                        print(f"⚠️ Error counting customers for emails_sent: {e}")
                        emails_sent = 0
            else:
                print(f"⚠️ Campaign {campaign_id} not found in database, will use fallback")
                
        except Exception as e:
            print(f"⚠️ Error fetching campaign details: {e}")
            emails_sent = 0
        
        # Update campaign status in database
        update_data = {
            "status": "completed",
            "completed_at": completed_at,
            "emails_sent": emails_sent  # Store calculated customer count as emails sent
        }
        
        # Update the campaign in database with retry logic
        try:
            result = supabase.table("campaign_logs")\
                .update(update_data)\
                .eq("campaign_id", campaign_id)\
                .execute()
            
            if not result.data:
                print(f"⚠️ Campaign {campaign_id} not found in database, creating new entry")
                # If campaign doesn't exist, create it
                campaign_info = {
                    "campaign_id": campaign_id,
                    "campaign_name": request_data.get("campaign_name", "Unknown Campaign"),
                    "segment_name": request_data.get("segment", "Unknown Segment"),
                    "product_name": request_data.get("product_name"),
                    "status": "completed",
                    "started_at": request_data.get("started_at", datetime.now().isoformat()),
                    "completed_at": update_data["completed_at"],
                    "emails_sent": update_data["emails_sent"],
                    "total_customers": request_data.get("total_customers", 0),
                    "success_rate": update_data.get("success_rate")
                }
                result = supabase.table("campaign_logs").insert(campaign_info).execute()
                
        except Exception as db_error:
            print(f"❌ Database error updating campaign {campaign_id}: {db_error}")
            # Continue execution even if database update fails
            print(f"⚠️ Campaign completion noted but not saved to database")
        
        print(f"✅ Campaign {campaign_id} marked as completed")
        
        return {
            "status": "success",
            "message": "Campaign completion notification processed",
            "campaign_id": campaign_id,
            "updated_data": update_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing campaign success webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process webhook: {str(e)}")

# --- Campaign Logging Functions ---

class CampaignStartRequest(BaseModel):
    campaign_name: str
    segment_name: str
    product_name: str = None
    total_customers: int = 0

class CampaignLogResponse(BaseModel):
    campaign_id: str
    campaign_name: str
    segment_name: str
    status: str
    started_at: str

def generate_campaign_id(campaign_name: str, segment_name: str) -> str:
    """Generate a unique campaign ID"""
    # Clean campaign name and segment name for ID
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', campaign_name.lower())
    clean_segment = re.sub(r'[^a-zA-Z0-9]', '_', segment_name.lower())
    timestamp = int(datetime.now().timestamp() * 1000)  # milliseconds
    
    return f"{clean_name}_{clean_segment}_{timestamp}"

@app.post("/campaigns/start")
async def start_campaign(request: CampaignStartRequest):
    """
    Start a new campaign and log it to database
    """
    try:
        # Generate campaign ID
        campaign_id = generate_campaign_id(request.campaign_name, request.segment_name)
        
        # Calculate actual customer count from database
        table_name = TABLE_NAME_MAP.get(request.segment_name)
        total_customers = 0
        
        if table_name:
            try:
                result = supabase.table(table_name).select("customer_id", count="exact").execute()
                total_customers = result.count if result.count is not None else 0
                print(f"📊 Calculated {total_customers} customers in {request.segment_name} segment")
            except Exception as e:
                print(f"⚠️ Error counting customers in {table_name}: {e}")
                total_customers = 0
        
        # Insert campaign log into database
        campaign_data = {
            "campaign_id": campaign_id,
            "campaign_name": request.campaign_name,
            "segment_name": request.segment_name,
            "product_name": request.product_name,
            "status": "running",
            "total_customers": total_customers,  # Use calculated count
            "started_at": datetime.now().isoformat()
        }
        
        result = supabase.table("campaign_logs").insert(campaign_data).execute()
        
        print(f"📊 Campaign started: {campaign_id}")
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "message": "Campaign started successfully",
            "data": campaign_data
        }
        
    except Exception as e:
        print(f"Error starting campaign: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start campaign: {str(e)}")

@app.get("/campaigns/recent")
async def get_recent_campaigns(limit: int = 10):
    """
    Get recent campaigns with their status
    """
    try:
        # Fetch recent campaigns from database
        result = supabase.table("campaign_logs")\
            .select("*")\
            .order("started_at", desc=True)\
            .limit(limit)\
            .execute()
        
        campaigns = result.data if result.data else []
        
        return {
            "success": True,
            "campaigns": campaigns,
            "total": len(campaigns)
        }
        
    except Exception as e:
        print(f"Error fetching recent campaigns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch campaigns: {str(e)}")

@app.get("/campaigns/{campaign_id}")
async def get_campaign_status(campaign_id: str):
    """
    Get status of a specific campaign
    """
    try:
        result = supabase.table("campaign_logs")\
            .select("*")\
            .eq("campaign_id", campaign_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        campaign = result.data[0]
        
        return {
            "success": True,
            "campaign": campaign
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching campaign status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch campaign: {str(e)}")

# --- Get Customer Counts Endpoint ---
@app.get("/segments/customer-counts")
async def get_customer_counts():
    """
    Get actual customer counts from database for each segment
    """
    try:
        customer_counts = {}
        
        for segment_label, table_name in TABLE_NAME_MAP.items():
            try:
                # Count customers in each segment table
                result = supabase.table(table_name).select("customer_id", count="exact").execute()
                customer_counts[segment_label] = result.count if result.count is not None else 0
                print(f"📊 {segment_label}: {customer_counts[segment_label]} customers")
            except Exception as e:
                print(f"⚠️ Error counting {segment_label}: {e}")
                customer_counts[segment_label] = 0
        
        return {
            "success": True,
            "customer_counts": customer_counts,
            "total_customers": sum(customer_counts.values())
        }
        
    except Exception as e:
        print(f"❌ Error fetching customer counts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch customer counts: {str(e)}")

# --- N8N Webhook Proxy Endpoint ---
@app.post("/trigger-n8n-campaign")
async def trigger_n8n_campaign(payload: dict):
    """
    Proxy endpoint to trigger n8n campaign workflow
    This avoids CORS issues by routing through our backend
    """
    try:
        print(f"📤 Proxying to n8n: {payload}")
        
        # Get n8n webhook URL from environment
        n8n_url = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook-test/campaign-trigger")
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                n8n_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            
        if response.status_code == 200:
            print("✅ Successfully triggered n8n campaign")
            return {
                "success": True,
                "message": "Campaign triggered successfully",
                "n8n_response": response.json() if response.content else {}
            }
        else:
            print(f"❌ n8n responded with status {response.status_code}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"n8n webhook failed: {response.text}"
            )
            
    except Exception as e:
        print(f"❌ Error triggering n8n campaign: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger n8n campaign: {str(e)}"
        )

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "healthy",
        "message": "Customer Segmentation & Contact API is running",
        "endpoints": {
            "segmentation": "/segment-and-store",
            "contacts": "/contacts",
            "campaign_success": "/campaign-success",
            "health": "/health"
        }
    }

# To run the server: uvicorn main:app --reload