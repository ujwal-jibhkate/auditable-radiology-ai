# 1. Switch to the more compatible 'slim' image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# 2. Use Debian's package manager 'apt-get'
RUN apt-get update && apt-get install -y --no-install-recommends supervisor && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for both services
# Combine into one layer to improve caching and reduce image size
RUN pip install --no-cache-dir -r backend/requirements.txt -r frontend/requirements.txt

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 3. Expose both ports (though only one will be accessible externally via Hugging Face)
EXPOSE 8000 8501

# The command to run when the container starts
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
