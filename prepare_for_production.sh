#!/bin/bash
set -eo pipefail

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
section() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Function to print success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning messages
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error messages
error() {
    echo -e "${RED}✗ $1${NC}"
}

# Main script starts here
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   JFKReveal Production Preparation      ${NC}"
echo -e "${BLUE}==========================================${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    warning "Virtual environment not found. Setting up..."
    make setup
else
    success "Virtual environment found"
fi

# Activate virtual environment
source venv/bin/activate

# 1. Run the production cleanup to remove unnecessary files
section "Cleaning Repository"
make production-clean
success "Repository cleaned"

# 2. Check for environment configuration
section "Environment Configuration"
if [ ! -f ".env" ]; then
    warning "No .env file found. Creating from example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        warning "Created .env from .env.example - PLEASE EDIT with your production credentials before deploying"
    else
        error "No .env.example file found. Please create a .env file manually."
    fi
else
    success "Environment file (.env) found"
fi

# 3. Run tests to ensure everything works
section "Running Tests"
echo -e "${BLUE}Running unit tests...${NC}"
make test-unit
UNIT_RESULT=$?

echo -e "${BLUE}Running integration tests...${NC}"
RUN_INTEGRATION=1 make test-integration
INTEGRATION_RESULT=$?

if [ -n "$RUN_E2E" ] && [ "$RUN_E2E" = "1" ]; then
    echo -e "${BLUE}Running end-to-end tests...${NC}"
    make test-e2e
    E2E_RESULT=$?
else
    warning "Skipping end-to-end tests. Set RUN_E2E=1 to run them."
    E2E_RESULT=0
fi

# Check if tests passed
if [ $UNIT_RESULT -eq 0 ] && [ $INTEGRATION_RESULT -eq 0 ] && [ $E2E_RESULT -eq 0 ]; then
    success "All tests passed"
else
    error "Tests failed. Please fix issues before proceeding to production"
    exit 1
fi

# 4. Build the package
section "Building Package"
make build
success "Package built successfully"

# 5. Summary and next steps
section "Summary"
echo -e "The JFKReveal package has been successfully prepared for production deployment!"
echo -e "\nNext steps:"
echo -e "1. Review the ${YELLOW}.env${NC} file to ensure all API keys and configuration are set properly"
echo -e "2. Deploy the package from the ${YELLOW}dist/${NC} directory"
echo -e "3. Set up monitoring and logging in your production environment"
echo -e "\nTo deploy the package, run:"
echo -e "   ${GREEN}pip install dist/jfkreveal-*.tar.gz${NC}"
echo -e "\n${BLUE}==========================================${NC}"

# Deactivate virtual environment
deactivate

exit 0 