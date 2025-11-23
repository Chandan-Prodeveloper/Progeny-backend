import os

# Bind to the PORT environment variable (required for Render)
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Worker configuration
workers = 1  # Keep low for free tier memory limits
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Longer timeout for model loading
keepalive = 5

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

# Process naming
proc_name = "progeny-backend"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Preload app for better performance (but uses more memory)
# Set to False for memory-constrained environments
preload_app = False