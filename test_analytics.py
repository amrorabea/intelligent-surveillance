#!/usr/bin/env python3

import sys
import os

# Add the streamlit directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'streamlit'))

try:
    from utils.api_client import SurveillanceAPIClient
    print("✅ Successfully imported SurveillanceAPIClient")
    
    # Create API client
    api_client = SurveillanceAPIClient()
    print(f"✅ API client created, base URL: {api_client.base_url}")
    print(f"✅ Auth token: {api_client.auth_token}")
    
    # Test analytics endpoint
    print("\n🔍 Testing analytics endpoint...")
    result = api_client.get_analytics()
    
    print("📊 Analytics result:")
    import json
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        print("✅ Analytics API working successfully!")
        data = result.get("data", {})
        print(f"📈 Total videos: {data.get('total_videos', 0)}")
        print(f"🎯 Total detections: {data.get('total_detections', 0)}")
    else:
        print(f"❌ Analytics API failed: {result.get('error')}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
