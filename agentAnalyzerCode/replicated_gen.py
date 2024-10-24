
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy import stats

app = FastAPI()

# Define data model for optimization request
class OptimizationRequest(BaseModel):
    initial_value: float

# Define a data model for integration request
class IntegrationRequest(BaseModel):
    lower_limit: float
    upper_limit: float

# Define a data model for statistics request
class StatisticsRequest(BaseModel):
    data: list[float]

@app.post('/optimize/')
def optimize(request: OptimizationRequest):
    """
    Endpoint to optimize a quadratic objective function.
    The objective function is f(x) = x^2 + 5x + 10.
    """
    # Objective function to minimize
    def objective(x):
        return x**2 + 5 * x + 10
    
    result = minimize(objective, request.initial_value)
    return {'optimal_value': result.x.tolist()}

@app.post('/integrate/')
def integrate(request: IntegrationRequest):
    """
    Endpoint to calculate the area under the curve of the function f(x) = x^2
    between the specified lower and upper limits using numerical integration.
    """
    # Integrand function
    def integrand(x):
        return x**2
    
    area, error = quad(integrand, request.lower_limit, request.upper_limit)
    return {'area_under_curve': area, 'error_estimate': error}

@app.post('/statistics/')
def statistics(request: StatisticsRequest):
    """
    Endpoint to calculate the mean and variance of a given list of numbers.
    """
    mean = stats.tmean(request.data)
    variance = stats.tvar(request.data)
    return {'mean': mean, 'variance': variance}

# To run the FastAPI app, use: uvicorn script_name:app --reload
