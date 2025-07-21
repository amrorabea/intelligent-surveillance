#!/usr/bin/env python3

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from controllers.VectorDBController import VectorDBController
    print("✅ Successfully imported VectorDBController")
    
    # Create vector controller
    vector_controller = VectorDBController()
    print(f"✅ VectorDBController created")
    print(f"📁 Collection name: {vector_controller.collection_name}")
    print(f"🔌 Is available: {vector_controller.is_available()}")
    
    # Try to get some data
    print("\n🔍 Testing semantic search...")
    results = vector_controller.semantic_search(
        query="object detection analysis surveillance",
        limit=10
    )
    
    print(f"📊 Search results count: {len(results)}")
    
    if results:
        print("✅ Found data in vector database!")
        for i, result in enumerate(results[:3]):  # Show first 3 results
            print(f"\n📄 Result {i+1}:")
            print(f"  ID: {result.get('id', 'N/A')}")
            metadata = result.get('metadata', {})
            print(f"  Project ID: {metadata.get('project_id', 'N/A')}")
            print(f"  Video: {metadata.get('video_filename', 'N/A')}")
            print(f"  Objects: {metadata.get('detected_objects', 'N/A')}")
            print(f"  Distance: {result.get('distance', 'N/A')}")
    else:
        print("❌ No data found in vector database")
        
        # Try to check what collections exist
        if vector_controller.client:
            try:
                collections = vector_controller.client.list_collections()
                print(f"\n📚 Available collections: {[c.name for c in collections]}")
                
                if collections:
                    for collection in collections:
                        count = collection.count()
                        print(f"  - {collection.name}: {count} documents")
            except Exception as e:
                print(f"❌ Error listing collections: {e}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
