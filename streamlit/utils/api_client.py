# streamlit/utils/api_client.py
import requests
import streamlit as st
from typing import Dict, Any, Optional, List
import time

class SurveillanceAPIClient:
    def __init__(self, base_url: str = "http://localhost:5000/api"):
        self.base_url = base_url
        self.auth_token = "dev"  # Use dev token for development
        
    def upload_file(self, project_id: str, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload a file to the surveillance system"""
        try:
            # Prepare the file for upload
            files = {
                "file": (filename, file_content, "video/mp4")  # Add MIME type
            }
            
            # Make the request
            response = requests.post(
                f"{self.base_url}/data/upload/{project_id}",
                files=files,
                timeout=300
            )
            
            # Debug information
            print(f"Upload URL: {response.url}")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:500]}")  # First 500 chars
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Handle different response formats from your backend
                if "signal" in response_data and response_data.get("signal") == "file_upload_success":
                    # Your backend returns this format
                    return {
                        "success": True,
                        "data": {
                            "file_id": response_data.get("file_id"),
                            "filename": filename,
                            "project_id": project_id
                        }
                    }
                elif "data" in response_data:
                    # Standard format
                    return {
                        "success": True,
                        "data": response_data["data"]
                    }
                else:
                    # Fallback - treat whole response as data
                    return {
                        "success": True,
                        "data": response_data
                    }
            else:
                # More detailed error handling
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return {"success": False, "error": error_msg}
                
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to backend server. Is it running on localhost:5000?"
            return {"success": False, "error": error_msg}
        except requests.exceptions.Timeout:
            error_msg = "Upload timed out. File might be too large."
            return {"success": False, "error": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"Upload failed: {str(e)}"
            return {"success": False, "error": error_msg}
    
    def process_video(self, project_id: str, file_id: str, **kwargs) -> Dict[str, Any]:
        """Start video processing job - try multiple endpoint patterns"""
        
        # List of possible endpoint patterns to try
        endpoints_to_try = [
            f"{self.base_url}/surveillance/process/{project_id}/{file_id}",  # Current attempt
            f"{self.base_url}/process/{project_id}/{file_id}",               # Alternative 1
            f"{self.base_url}/surveillance/analyze/{project_id}/{file_id}",  # Alternative 2
            f"{self.base_url}/video/process/{project_id}/{file_id}",         # Alternative 3
            f"{self.base_url}/data/process/{project_id}/{file_id}",          # Alternative 4
        ]
        
        # Prepare processing parameters
        process_data = {
            "sample_rate": kwargs.get("sample_rate", 1.0),
            "detection_threshold": kwargs.get("detection_threshold", 0.5),
            "enable_tracking": kwargs.get("enable_tracking", True),
            "enable_captioning": kwargs.get("enable_captioning", True)
        }
        
        # Try each endpoint until one works
        for endpoint_url in endpoints_to_try:
            try:
                print(f"Trying endpoint: {endpoint_url}")
                print(f"Process Data: {process_data}")
                
                response = requests.post(
                    endpoint_url,
                    json=process_data,
                    timeout=30
                )
                
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Handle different response formats
                    if "data" in response_data:
                        return {
                            "success": True,
                            "data": response_data["data"]
                        }
                    else:
                        return {
                            "success": True,
                            "data": response_data
                        }
                elif response.status_code == 404:
                    # Try next endpoint
                    continue
                else:
                    # Other error, return it
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    return {"success": False, "error": error_msg}
                    
            except requests.exceptions.RequestException as e:
                # Try next endpoint
                continue
        
        # If all endpoints failed
        return {
            "success": False, 
            "error": f"All process endpoints failed. Tried: {endpoints_to_try}"
        }
    
    def get_available_endpoints(self) -> Dict[str, Any]:
        """Get list of available API endpoints"""
        try:
            # Try the OpenAPI/docs endpoint
            response = requests.get(f"{self.base_url.replace('/api', '')}/docs", timeout=5)
            if response.status_code == 200:
                return {"success": True, "info": "Check /docs for available endpoints"}
            
            # Try health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            
            # Try root API endpoint
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
                
            return {"success": False, "error": "No endpoints responded"}
            
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status - try multiple endpoint patterns"""
        
        endpoints_to_try = [
            f"{self.base_url}/surveillance/jobs/status/{job_id}",  # Current
            f"{self.base_url}/jobs/status/{job_id}",               # Alternative 1
            f"{self.base_url}/surveillance/job/{job_id}",          # Alternative 2
            f"{self.base_url}/job/{job_id}/status",                # Alternative 3
            f"{self.base_url}/status/{job_id}",                    # Alternative 4
        ]
        
        for endpoint_url in endpoints_to_try:
            try:
                response = requests.get(endpoint_url, timeout=10)
                
                if response.status_code == 200:
                    return {"success": True, "status": response.json()}
                elif response.status_code == 404:
                    continue  # Try next endpoint
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                    
            except requests.exceptions.RequestException:
                continue  # Try next endpoint
        
        return {"success": False, "error": f"All job status endpoints failed. Tried: {endpoints_to_try}"}
    
    def semantic_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform semantic search with detailed debugging"""
        
        # Clean and validate parameters
        params = {
            "query": str(query).strip() if query else "",
            "max_results": int(kwargs.get("max_results", 10)),
            "confidence_threshold": float(kwargs.get("confidence_threshold", 0.3)) if kwargs.get("confidence_threshold") else 0.3,
        }
        
        # Add optional parameters only if they have valid values
        if kwargs.get("project_id") and str(kwargs.get("project_id")).strip():
            params["project_id"] = str(kwargs.get("project_id")).strip()
            
        if kwargs.get("start_date") and str(kwargs.get("start_date")).strip():
            params["start_date"] = str(kwargs.get("start_date")).strip()
            
        if kwargs.get("end_date") and str(kwargs.get("end_date")).strip():
            params["end_date"] = str(kwargs.get("end_date")).strip()
            
        if kwargs.get("object_types") and len(kwargs.get("object_types", [])) > 0:
            params["object_types"] = kwargs.get("object_types")
        
        # Validate query
        if not params["query"]:
            return {"success": False, "error": "Query cannot be empty"}
        
        # Try the main endpoint
        endpoint_url = f"{self.base_url}/surveillance/query"
        
        # Prepare headers with authentication
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        try:
            print(f"🔍 Making search request to: {endpoint_url}")
            print(f"📋 Parameters: {params}")
            
            response = requests.get(endpoint_url, params=params, headers=headers, timeout=30)
            
            print(f"📡 Response Status: {response.status_code}")
            print(f"📄 Response Headers: {dict(response.headers)}")
            print(f"📝 Response Body: {response.text}")
            
            if response.status_code == 200:
                response_data = response.json()
                return {"success": True, "data": response_data}
            else:
                # Return detailed error information
                return {
                    "success": False, 
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": endpoint_url,
                    "params": params
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False, 
                "error": f"Request failed: {str(e)}",
                "url": endpoint_url,
                "params": params
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics using available endpoints"""
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            
            # Try health endpoint first (this exists in the backend)
            response = requests.get(f"{self.base_url}/surveillance/health", headers=headers, timeout=10)
            if response.status_code == 200:
                return {"success": True, "stats": response.json()}
            
            # If health endpoint fails, try analytics endpoint
            response = requests.get(f"{self.base_url}/surveillance/analytics", headers=headers, timeout=10)
            if response.status_code == 200:
                return {"success": True, "stats": response.json()}
            
            # If both fail but we get a 404, server is up but endpoints aren't implemented
            if response.status_code == 404:
                return {"success": True, "stats": {"message": "Backend is running but endpoints not fully implemented"}}
            
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def test_search_fix(self) -> Dict[str, Any]:
        """Quick test to verify the search fix"""
        try:
            # Test with minimal parameters
            test_params = {"query": "test", "max_results": 1}
            response = requests.get(f"{self.base_url}/surveillance/query", params=test_params, timeout=10)
            
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.text[:500] if response.text else "Empty response"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_frame(self, result_id: str = None, frame_path: str = None, project_id: str = None) -> Optional[bytes]:
        """Get frame image data from the backend"""
        
        # Try the direct frame endpoint first (most reliable)
        if result_id:
            try:
                direct_url = f"{self.base_url}/surveillance/frame-direct/{result_id}"
                params = {"project_id": project_id or "default"}
                
                response = requests.get(direct_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type.lower():
                        return response.content
                        
            except requests.exceptions.RequestException:
                pass
        
        # Fallback to other endpoints
        endpoints_to_try = []
        
        if result_id:
            endpoints_to_try.extend([
                f"{self.base_url}/surveillance/frame/{result_id}",
                f"{self.base_url}/surveillance/frames/{result_id}",
            ])
        
        if frame_path and project_id:
            endpoints_to_try.extend([
                f"{self.base_url}/surveillance/surveillance/projects/{project_id}/frame",
                f"{self.base_url}/surveillance/data/frame/{project_id}",
            ])
        
        # Try each endpoint
        for endpoint_url in endpoints_to_try:
            try:
                params = {}
                if frame_path:
                    params["frame_path"] = frame_path
                if project_id:
                    params["project_id"] = project_id
                
                response = requests.get(endpoint_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    # Check if response is an image
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type.lower():
                        return response.content
                    else:
                        # Might be JSON with base64 data
                        try:
                            data = response.json()
                            if 'image_data' in data:
                                import base64
                                return base64.b64decode(data['image_data'])
                        except Exception:
                            pass
                elif response.status_code == 404:
                    continue  # Try next endpoint
                    
            except requests.exceptions.RequestException:
                continue  # Try next endpoint
        
        # If all methods fail, return None
        return None

    def get_frame_from_metadata(self, metadata: Dict[str, Any]) -> Optional[bytes]:
        """Extract frame using metadata information"""
        
        # Try different metadata keys for frame information
        result_id = metadata.get('result_id') or metadata.get('id')
        frame_path = metadata.get('frame_path') or metadata.get('file_path')
        project_id = metadata.get('project_id')
        
        return self.get_frame(result_id=result_id, frame_path=frame_path, project_id=project_id)
    
    def check_frame_endpoints(self) -> Dict[str, Any]:
        """Check what frame-related endpoints are available"""
        
        test_endpoints = [
            f"{self.base_url}/surveillance/frame/test",
            f"{self.base_url}/frames/test", 
            f"{self.base_url}/frame",
            f"{self.base_url}/surveillance/frames",
            f"{self.base_url}/data/frame",
        ]
        
        available_endpoints = []
        
        for endpoint in test_endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                # Even 404 means the endpoint exists but the resource doesn't
                if response.status_code in [200, 404, 405]:  # 405 = Method not allowed
                    available_endpoints.append({
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "note": "Available" if response.status_code == 200 else 
                               "Endpoint exists" if response.status_code in [404, 405] else "Unknown"
                    })
            except Exception:
                pass
        
        return {
            "available_endpoints": available_endpoints,
            "total_found": len(available_endpoints)
        }
    
    def get_analytics(self, heavy_mode: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Get analytics data from the surveillance system
        
        Args:
            heavy_mode (bool): If True, use heavy analytics (loads sentence transformer).
                              If False, use light analytics (no model loading).
            **kwargs: Additional parameters like project_id, date filters, etc.
        """
        
        # Prepare parameters
        params = {}
        
        # Add time filters if provided
        if kwargs.get("start_date"):
            params["start_date"] = kwargs.get("start_date")
        if kwargs.get("end_date"):
            params["end_date"] = kwargs.get("end_date")
        if kwargs.get("project_id"):
            params["project_id"] = kwargs.get("project_id")
        if kwargs.get("time_range"):
            params["time_range"] = kwargs.get("time_range")
        
        # Choose endpoints based on mode
        if heavy_mode:
            # Heavy analytics endpoints (loads sentence transformer model)
            endpoints_to_try = [
                f"{self.base_url}/surveillance/analytics",  # Main heavy analytics
                f"{self.base_url}/analytics", 
                f"{self.base_url}/surveillance/analytics/summary",
            ]
            
            # If project_id is specified, try project-specific endpoints
            if params.get("project_id"):
                project_endpoints = [
                    f"{self.base_url}/surveillance/analytics/summary/{params['project_id']}",
                    f"{self.base_url}/surveillance/analytics/{params['project_id']}",
                    f"{self.base_url}/analytics/{params['project_id']}",
                ]
                endpoints_to_try = project_endpoints + endpoints_to_try
        else:
            # Light analytics endpoints (no model loading)
            endpoints_to_try = [
                f"{self.base_url}/surveillance/analytics/light",  # New light endpoint
                f"{self.base_url}/surveillance/health",  # Fallback to health
            ]
        
        # Prepare headers with authentication
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        for endpoint_url in endpoints_to_try:
            try:
                # Use longer timeout for heavy mode
                timeout = 45 if heavy_mode else 15
                print(f"🔍 Requesting: {endpoint_url}")
                print(f"📋 Headers: {headers}")
                print(f"📋 Params: {params}")
                response = requests.get(endpoint_url, params=params, headers=headers, timeout=timeout)
                print(f"📡 Response Status: {response.status_code}")
                print(f"📝 Response: {response.text[:300]}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Handle different response formats
                    if "data" in response_data:
                        return {"success": True, "data": response_data["data"]}
                    elif "analytics" in response_data:
                        return {"success": True, "data": response_data["analytics"]}
                    else:
                        return {"success": True, "data": response_data}
                        
                elif response.status_code == 404:
                    continue  # Try next endpoint
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    return {"success": False, "error": error_msg}
                    
            except requests.exceptions.RequestException:
                continue  # Try next endpoint
        
        mode_desc = "heavy" if heavy_mode else "light"
        return {
            "success": False, 
            "error": f"All {mode_desc} analytics endpoints failed. Tried: {endpoints_to_try}"
        }

    def get_surveillance_stats(self, project_id: str = None) -> Dict[str, Any]:
        """Get surveillance statistics using available endpoints (lightweight version)"""
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            
            # Only use the health endpoint to avoid triggering VectorDB loading
            # The analytics endpoint causes sentence transformer loading which is too heavy
            endpoint = f"{self.base_url}/surveillance/health"
            response = requests.get(endpoint, headers=headers, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                # Transform health data into stats format if needed
                if isinstance(health_data, dict):
                    return {"success": True, "stats": health_data}
                else:
                    return {"success": True, "stats": {"health": health_data}}
            else:
                return {"success": False, "error": f"Health endpoint returned HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Simple health check to see if the backend is running"""
        try:
            # Try the base URL first
            response = requests.get("http://localhost:5000", timeout=5)
            if response.status_code in [200, 404]:  # Either is fine, means server is up
                return {"success": True, "status": "Backend is running"}
            else:
                return {"success": False, "error": f"Unexpected status code: {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Cannot connect to backend"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Backend timeout"}
        except Exception as e:
            return {"success": False, "error": f"Health check failed: {str(e)}"}
    
    def discover_projects(self) -> Dict[str, Any]:
        """Attempt to discover available projects by analyzing database content"""
        try:
            # First try to get general database stats
            stats_result = self.get_database_stats()
            if not stats_result.get("success", False):
                return {"success": False, "error": "Database not accessible"}
            
            discovered_projects = []
            
            # Try common project patterns with test searches
            common_patterns = [
                "test-project", "surveillance-demo", "camera-01", "security-main",
                "lobby-security", "parking-surveillance", "entrance-cam",
                "surveillance-2024", "camera-main", "security-test"
            ]
            
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            
            for pattern in common_patterns:
                try:
                    # Quick test search to see if project has data
                    params = {
                        "query": "test", 
                        "project_id": pattern,
                        "max_results": 1,
                        "confidence_threshold": 0.1
                    }
                    
                    response = requests.get(
                        f"{self.base_url}/surveillance/query", 
                        params=params, 
                        headers=headers, 
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        results = response_data.get("results", [])
                        if len(results) > 0:
                            discovered_projects.append({
                                "project_id": pattern,
                                "source": "pattern_test",
                                "type": "verified",
                                "data_count": len(results)
                            })
                except Exception:
                    continue
            
            return {
                "success": True,
                "projects": discovered_projects,
                "total_discovered": len(discovered_projects)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Project discovery failed: {str(e)}"
            }

# Global API client instance
api_client = SurveillanceAPIClient()