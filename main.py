"""
Agricultural Yield Predictor
AI-powered crop yield prediction based on weather and conditions
Author: Pranay M
"""

import ollama
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys

console = Console()
MODEL = "llama3.2"
DATA_DIR = Path("agriculture_data")
DATA_DIR.mkdir(exist_ok=True)

# Common crops database
CROPS = {
    "wheat": {"season": "winter", "growing_days": 120, "optimal_temp": (15, 25), "water_needs": "moderate"},
    "corn": {"season": "summer", "growing_days": 90, "optimal_temp": (20, 30), "water_needs": "high"},
    "rice": {"season": "monsoon", "growing_days": 120, "optimal_temp": (20, 35), "water_needs": "very_high"},
    "soybeans": {"season": "summer", "growing_days": 100, "optimal_temp": (20, 30), "water_needs": "moderate"},
    "cotton": {"season": "summer", "growing_days": 150, "optimal_temp": (25, 35), "water_needs": "moderate"},
    "potatoes": {"season": "spring", "growing_days": 90, "optimal_temp": (15, 20), "water_needs": "moderate"},
    "tomatoes": {"season": "summer", "growing_days": 80, "optimal_temp": (20, 30), "water_needs": "high"},
    "sugarcane": {"season": "tropical", "growing_days": 365, "optimal_temp": (25, 35), "water_needs": "very_high"}
}


class WeatherAnalyzer:
    """Analyze weather data for agriculture"""
    
    def analyze_conditions(self, weather_data: dict) -> str:
        """Analyze weather conditions for farming"""
        prompt = f"""Analyze these weather conditions for agricultural purposes:

WEATHER DATA:
{json.dumps(weather_data, indent=2)}

Analyze:
1. **Temperature Analysis**: How suitable for crops?
2. **Precipitation**: Adequate, excess, or deficit?
3. **Growing Degree Days**: Accumulated heat units
4. **Frost Risk**: Any freeze concerns?
5. **Drought Indicators**: Moisture stress signs?
6. **Storm Risk**: Severe weather concerns?

Provide agricultural recommendations based on conditions."""

        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']
    
    def forecast_impact(self, forecast: dict, crop: str) -> str:
        """Predict weather impact on specific crop"""
        crop_info = CROPS.get(crop.lower(), {})
        
        prompt = f"""Predict how this weather forecast will impact {crop}:

CROP INFO:
{json.dumps(crop_info, indent=2)}

WEATHER FORECAST:
{json.dumps(forecast, indent=2)}

Analyze:
1. **Growth Impact**: How will growth be affected?
2. **Yield Prediction**: Expected yield impact (% change)
3. **Stress Factors**: What stresses will the crop face?
4. **Critical Periods**: Any critical growth stage concerns?
5. **Mitigation**: What can farmers do to protect crops?

Be specific with recommendations."""

        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']


class YieldPredictor:
    """Predict agricultural yields"""
    
    def predict_yield(self, farm_data: dict) -> dict:
        """Predict crop yield"""
        prompt = f"""Predict crop yield based on this data:

FARM DATA:
- Crop: {farm_data.get('crop', 'Unknown')}
- Field Size: {farm_data.get('area_hectares', 0)} hectares
- Soil Type: {farm_data.get('soil_type', 'Unknown')}
- Irrigation: {farm_data.get('irrigation', 'Unknown')}
- Fertilizer Used: {farm_data.get('fertilizer', 'Unknown')}
- Planting Date: {farm_data.get('planting_date', 'Unknown')}
- Current Growth Stage: {farm_data.get('growth_stage', 'Unknown')}
- Weather Summary: {farm_data.get('weather_summary', 'Unknown')}
- Historical Yield: {farm_data.get('historical_yield', 'Unknown')} tons/hectare

Provide:
1. **Predicted Yield**: Tons per hectare (with confidence range)
2. **Total Expected**: Total production
3. **Comparison**: vs historical average
4. **Key Factors**: What's driving the prediction
5. **Risks**: What could change the outcome
6. **Optimization**: How to maximize yield

Format yield predictions as JSON with: predicted_yield, confidence_low, confidence_high, total_expected, risk_factors"""

        response = ollama.generate(model=MODEL, prompt=prompt)
        
        try:
            text = response['response']
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        
        return {"analysis": response['response']}


class SoilAnalyzer:
    """Analyze soil conditions"""
    
    def analyze_soil(self, soil_data: dict) -> str:
        """Analyze soil for crop suitability"""
        prompt = f"""Analyze this soil data for agriculture:

SOIL DATA:
{json.dumps(soil_data, indent=2)}

Analyze:
1. **Soil Health Score**: Overall rating 1-10
2. **Nutrient Status**: N, P, K levels assessment
3. **pH Analysis**: Is it optimal? Amendments needed?
4. **Organic Matter**: Status and improvement suggestions
5. **Drainage**: Water retention characteristics
6. **Recommended Crops**: Best suited crops
7. **Improvement Plan**: How to enhance soil health

Provide specific, actionable recommendations."""

        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']
    
    def recommend_amendments(self, soil_data: dict, target_crop: str) -> str:
        """Recommend soil amendments for specific crop"""
        prompt = f"""Recommend soil amendments for growing {target_crop}:

CURRENT SOIL:
{json.dumps(soil_data, indent=2)}

CROP REQUIREMENTS:
{json.dumps(CROPS.get(target_crop.lower(), {}), indent=2)}

Provide:
1. **Required Amendments**: What to add
2. **Application Rates**: How much per hectare
3. **Timing**: When to apply
4. **Method**: How to apply
5. **Cost Estimate**: Approximate costs
6. **Expected Improvement**: Results timeline"""

        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']


class CropAdvisor:
    """Advise on crop selection and management"""
    
    def recommend_crops(self, conditions: dict) -> str:
        """Recommend crops based on conditions"""
        prompt = f"""Recommend the best crops for these conditions:

CONDITIONS:
- Location/Climate: {conditions.get('climate', 'Unknown')}
- Soil Type: {conditions.get('soil', 'Unknown')}
- Water Availability: {conditions.get('water', 'Unknown')}
- Season: {conditions.get('season', 'Unknown')}
- Labor Available: {conditions.get('labor', 'Unknown')}
- Market Access: {conditions.get('market', 'Unknown')}
- Budget: {conditions.get('budget', 'Unknown')}

Recommend:
1. **Primary Crop**: Best main crop with reasoning
2. **Secondary Options**: 2-3 alternatives
3. **Companion Planting**: What to plant together
4. **Rotation Plan**: Multi-season strategy
5. **Risk Assessment**: Potential challenges
6. **Expected Returns**: Profit potential

Rank by profitability and feasibility."""

        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']
    
    def get_growing_guide(self, crop: str) -> str:
        """Get comprehensive growing guide"""
        crop_info = CROPS.get(crop.lower(), {})
        
        prompt = f"""Provide a comprehensive growing guide for {crop}:

CROP INFO:
{json.dumps(crop_info, indent=2)}

Include:
1. **Site Preparation**: How to prepare the field
2. **Planting**: When, how, spacing, depth
3. **Fertilization Schedule**: What, when, how much
4. **Irrigation Management**: Water requirements by stage
5. **Pest Management**: Common pests and controls
6. **Disease Prevention**: Common diseases and prevention
7. **Weed Control**: Management strategies
8. **Growth Stages**: Key milestones to watch
9. **Harvest**: When and how to harvest
10. **Post-Harvest**: Storage and handling

Make it practical for farmers."""

        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']


class PestDiseaseAdvisor:
    """Advise on pest and disease management"""
    
    def diagnose(self, symptoms: str, crop: str) -> str:
        """Diagnose crop problems"""
        prompt = f"""Diagnose this crop problem:

CROP: {crop}
SYMPTOMS: {symptoms}

Provide:
1. **Likely Diagnosis**: Most probable cause(s)
2. **Confirmation**: How to confirm diagnosis
3. **Severity Assessment**: How serious is it?
4. **Immediate Actions**: What to do now
5. **Treatment Options**: Both organic and conventional
6. **Prevention**: How to prevent recurrence
7. **Spread Risk**: Will it affect other plants?

Prioritize integrated pest management approaches."""

        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']


class FarmRecordManager:
    """Manage farm records"""
    
    def __init__(self):
        self.records_file = DATA_DIR / "farm_records.json"
        self.records = self._load()
    
    def _load(self) -> dict:
        if self.records_file.exists():
            return json.loads(self.records_file.read_text())
        return {"fields": {}, "harvests": [], "inputs": []}
    
    def _save(self):
        self.records_file.write_text(json.dumps(self.records, indent=2))
    
    def add_field(self, name: str, data: dict):
        self.records["fields"][name] = {**data, "created": datetime.now().isoformat()}
        self._save()
    
    def log_input(self, field: str, input_type: str, amount: str, notes: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "field": field,
            "type": input_type,
            "amount": amount,
            "notes": notes
        }
        self.records["inputs"].append(entry)
        self._save()
    
    def log_harvest(self, field: str, crop: str, yield_amount: float, notes: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "field": field,
            "crop": crop,
            "yield": yield_amount,
            "notes": notes
        }
        self.records["harvests"].append(entry)
        self._save()
    
    def get_field_history(self, field: str) -> dict:
        inputs = [i for i in self.records["inputs"] if i["field"] == field]
        harvests = [h for h in self.records["harvests"] if h["field"] == field]
        return {"field": self.records["fields"].get(field, {}), "inputs": inputs, "harvests": harvests}


class MarketAnalyzer:
    """Analyze agricultural markets"""
    
    def analyze_market(self, crop: str, region: str = "") -> str:
        """Analyze market conditions"""
        prompt = f"""Analyze market conditions for {crop}:

REGION: {region if region else 'General'}

Analyze:
1. **Current Prices**: Typical price ranges
2. **Demand Outlook**: Market demand trends
3. **Supply Factors**: What affects supply
4. **Best Selling Time**: Optimal time to sell
5. **Market Channels**: Where to sell
6. **Value Addition**: Processing opportunities
7. **Contract Farming**: Opportunities available
8. **Export Potential**: International markets

Provide practical market advice for farmers."""

        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']


# ============= CLI Interface =============

def show_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          🌾 Agricultural Yield Predictor 🌾                   ║
║           AI-Powered Farming Intelligence                     ║
║                   Author: Pranay M                            ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold green"))


def show_menu():
    table = Table(title="Farm Tools", show_header=False, box=None)
    table.add_column("Option", style="cyan")
    table.add_column("Description")
    
    table.add_row("1", "🌱 Predict Crop Yield")
    table.add_row("2", "🌤️  Weather Impact Analysis")
    table.add_row("3", "🪴 Soil Analysis")
    table.add_row("4", "📋 Crop Recommendations")
    table.add_row("5", "📖 Growing Guide")
    table.add_row("6", "🐛 Pest/Disease Diagnosis")
    table.add_row("7", "📊 Market Analysis")
    table.add_row("8", "📁 Farm Records")
    table.add_row("9", "🌾 Crop Database")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)


def predict_yield():
    """Predict crop yield"""
    console.print("\n[cyan]Yield Prediction[/cyan]\n")
    
    farm_data = {
        "crop": Prompt.ask("Crop", default="wheat"),
        "area_hectares": float(Prompt.ask("Field size (hectares)", default="10")),
        "soil_type": Prompt.ask("Soil type", default="loamy"),
        "irrigation": Prompt.ask("Irrigation type", default="drip"),
        "fertilizer": Prompt.ask("Fertilizer used", default="NPK balanced"),
        "planting_date": Prompt.ask("Planting date", default="2024-03-01"),
        "growth_stage": Prompt.ask("Current growth stage", default="flowering"),
        "weather_summary": Prompt.ask("Weather summary", default="normal rainfall, moderate temperatures"),
        "historical_yield": Prompt.ask("Historical yield (tons/ha)", default="3.5")
    }
    
    predictor = YieldPredictor()
    
    with Progress(SpinnerColumn(), TextColumn("Predicting yield...")) as progress:
        task = progress.add_task("", total=None)
        result = predictor.predict_yield(farm_data)
    
    if "analysis" in result:
        console.print(Panel(result["analysis"], title="Yield Prediction"))
    else:
        table = Table(title="Yield Prediction")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Predicted Yield", f"{result.get('predicted_yield', 'N/A')} tons/ha")
        table.add_row("Confidence Range", f"{result.get('confidence_low', 'N/A')} - {result.get('confidence_high', 'N/A')}")
        table.add_row("Total Expected", f"{result.get('total_expected', 'N/A')} tons")
        
        console.print(table)


def weather_analysis():
    """Analyze weather impact"""
    crop = Prompt.ask("Crop to analyze", default="wheat")
    
    # Simulated weather data
    weather = {
        "current_temp": Prompt.ask("Current temperature (°C)", default="22"),
        "rainfall_last_week": Prompt.ask("Rainfall last week (mm)", default="25"),
        "forecast_temp": Prompt.ask("Forecast temperature (°C)", default="24"),
        "forecast_rainfall": Prompt.ask("Expected rainfall next week (mm)", default="30"),
        "humidity": Prompt.ask("Humidity (%)", default="65")
    }
    
    analyzer = WeatherAnalyzer()
    
    with Progress(SpinnerColumn(), TextColumn("Analyzing...")) as progress:
        task = progress.add_task("", total=None)
        impact = analyzer.forecast_impact(weather, crop)
    
    console.print(Panel(impact, title="Weather Impact Analysis"))


def soil_analysis():
    """Analyze soil conditions"""
    soil_data = {
        "ph": Prompt.ask("Soil pH", default="6.5"),
        "nitrogen": Prompt.ask("Nitrogen level", default="medium"),
        "phosphorus": Prompt.ask("Phosphorus level", default="low"),
        "potassium": Prompt.ask("Potassium level", default="medium"),
        "organic_matter": Prompt.ask("Organic matter (%)", default="3"),
        "texture": Prompt.ask("Soil texture", default="loamy"),
        "drainage": Prompt.ask("Drainage", default="good")
    }
    
    analyzer = SoilAnalyzer()
    
    with Progress(SpinnerColumn(), TextColumn("Analyzing soil...")) as progress:
        task = progress.add_task("", total=None)
        analysis = analyzer.analyze_soil(soil_data)
    
    console.print(Panel(analysis, title="Soil Analysis"))


def crop_recommendations():
    """Get crop recommendations"""
    conditions = {
        "climate": Prompt.ask("Climate/Location", default="temperate"),
        "soil": Prompt.ask("Soil type", default="loamy"),
        "water": Prompt.ask("Water availability", default="moderate"),
        "season": Prompt.ask("Season", default="spring"),
        "labor": Prompt.ask("Labor availability", default="moderate"),
        "market": Prompt.ask("Market access", default="good"),
        "budget": Prompt.ask("Budget", default="moderate")
    }
    
    advisor = CropAdvisor()
    
    with Progress(SpinnerColumn(), TextColumn("Analyzing...")) as progress:
        task = progress.add_task("", total=None)
        recommendations = advisor.recommend_crops(conditions)
    
    console.print(Panel(recommendations, title="Crop Recommendations"))


def growing_guide():
    """Get growing guide"""
    console.print("\n[cyan]Available crops:[/cyan]")
    for crop in CROPS.keys():
        console.print(f"  • {crop}")
    
    crop = Prompt.ask("\nSelect crop", default="wheat")
    
    advisor = CropAdvisor()
    
    with Progress(SpinnerColumn(), TextColumn("Generating guide...")) as progress:
        task = progress.add_task("", total=None)
        guide = advisor.get_growing_guide(crop)
    
    console.print(Panel(guide, title=f"Growing Guide: {crop.title()}"))


def pest_diagnosis():
    """Diagnose pest/disease"""
    crop = Prompt.ask("Affected crop")
    symptoms = Prompt.ask("Describe symptoms")
    
    advisor = PestDiseaseAdvisor()
    
    with Progress(SpinnerColumn(), TextColumn("Diagnosing...")) as progress:
        task = progress.add_task("", total=None)
        diagnosis = advisor.diagnose(symptoms, crop)
    
    console.print(Panel(diagnosis, title="Diagnosis & Treatment"))


def market_analysis():
    """Analyze market"""
    crop = Prompt.ask("Crop")
    region = Prompt.ask("Region (optional)", default="")
    
    analyzer = MarketAnalyzer()
    
    with Progress(SpinnerColumn(), TextColumn("Analyzing market...")) as progress:
        task = progress.add_task("", total=None)
        analysis = analyzer.analyze_market(crop, region)
    
    console.print(Panel(analysis, title="Market Analysis"))


def farm_records():
    """Manage farm records"""
    manager = FarmRecordManager()
    
    console.print("\n[cyan]Farm Records[/cyan]")
    console.print("1. Add field")
    console.print("2. Log input (fertilizer/pesticide)")
    console.print("3. Log harvest")
    console.print("4. View field history")
    
    choice = Prompt.ask("Select", default="4")
    
    if choice == "1":
        name = Prompt.ask("Field name")
        data = {
            "size_hectares": float(Prompt.ask("Size (hectares)")),
            "soil_type": Prompt.ask("Soil type"),
            "current_crop": Prompt.ask("Current crop", default="")
        }
        manager.add_field(name, data)
        console.print(f"[green]Field '{name}' added[/green]")
    
    elif choice == "2":
        field = Prompt.ask("Field name")
        input_type = Prompt.ask("Input type (fertilizer/pesticide/seed)")
        amount = Prompt.ask("Amount applied")
        notes = Prompt.ask("Notes", default="")
        manager.log_input(field, input_type, amount, notes)
        console.print("[green]Input logged[/green]")
    
    elif choice == "3":
        field = Prompt.ask("Field name")
        crop = Prompt.ask("Crop harvested")
        yield_amt = float(Prompt.ask("Yield (tons)"))
        notes = Prompt.ask("Notes", default="")
        manager.log_harvest(field, crop, yield_amt, notes)
        console.print("[green]Harvest logged[/green]")
    
    elif choice == "4":
        field = Prompt.ask("Field name")
        history = manager.get_field_history(field)
        console.print(Panel(json.dumps(history, indent=2), title=f"History: {field}"))


def show_crop_database():
    """Display crop database"""
    table = Table(title="Crop Database")
    table.add_column("Crop", style="cyan")
    table.add_column("Season")
    table.add_column("Growing Days")
    table.add_column("Optimal Temp (°C)")
    table.add_column("Water Needs")
    
    for crop, info in CROPS.items():
        temp_range = f"{info['optimal_temp'][0]}-{info['optimal_temp'][1]}"
        table.add_row(
            crop.title(),
            info['season'],
            str(info['growing_days']),
            temp_range,
            info['water_needs']
        )
    
    console.print(table)


def main():
    show_banner()
    
    try:
        ollama.list()
    except Exception:
        console.print("[red]Error: Ollama not running. Start with: ollama serve[/red]")
        sys.exit(1)
    
    while True:
        show_menu()
        choice = Prompt.ask("\nSelect option", default="0")
        
        actions = {
            "1": predict_yield, "2": weather_analysis, "3": soil_analysis,
            "4": crop_recommendations, "5": growing_guide, "6": pest_diagnosis,
            "7": market_analysis, "8": farm_records, "9": show_crop_database
        }
        
        if choice == "0":
            console.print("[yellow]Happy farming! 🌾[/yellow]")
            break
        elif choice in actions:
            actions[choice]()
        else:
            console.print("[red]Invalid option[/red]")
        
        console.print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
