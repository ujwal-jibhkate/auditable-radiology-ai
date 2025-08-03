# 1. Switch from the Debian-based 'slim' to the more secure 'alpine'
FROM python:3.11-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy all your project files into the container
COPY . .

# 2. Use Alpine's package manager 'apk' instead of 'apt-get'
RUN apk add --no-cache supervisor

# Install Python dependencies for both backend and frontend
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the ports for the backend and frontend
EXPOSE 8000 8501

# The command to run when the container starts
CMD ["/usr/bin/supervisord"]