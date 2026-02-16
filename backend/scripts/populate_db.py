import sys
import os
import urllib.request
import time
from pathlib import Path

# Add parent directory to path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ExternalMemorySystem

def download_rfc(rfc_number):
    url = f"https://www.rfc-editor.org/rfc/rfc{rfc_number}.txt"
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download RFC {rfc_number}: {e}")
        return None

def populate():
    # List of significant RFCs (Standards)
    rfcs = [
        # HTTP
        2616, # HTTP/1.1 (Obsolete but large)
        7540, # HTTP/2
        
        # Email
        5321, # SMTP
        5322, # Internet Message Format
        
        # Core Internet
        791,  # IP
        793,  # TCP
        
        # Security
        5246, # TLS 1.2
        8446, # TLS 1.3
        
        # Data Formats
        8259, # JSON
        4180, # CSV
        
        # Others
        3986, # URI Generic Syntax
        1035, # Domain Names
    ]
    
    print("Initializing Memory System...")
    # Initialize with existing config
    system = ExternalMemorySystem(config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    
    print(f"Starting ingestion of {len(rfcs)} RFC documents...")
    
    total_chunks = 0
    
    for rfc_num in rfcs:
        print(f"Downloading RFC {rfc_num}...", end="", flush=True)
        text = download_rfc(rfc_num)
        
        if text:
            print(f" Ingesting...", end="", flush=True)
            stats = system.ingest_document(
                text, 
                source=f"RFC {rfc_num}", 
                file_type="technical_standard",
                extension="txt"
            )
            print(f" Done. ({stats['chunks_created']} chunks)")
            total_chunks += stats['chunks_created']
        else:
            print(" Skipped.")
            
        # Be nice to the IETF server
        time.sleep(1.0)
        
    print(f"\nPopulation complete! Added {total_chunks} chunks to the knowledge base.")

if __name__ == "__main__":
    populate()
