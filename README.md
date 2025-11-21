# Fridge API Endpoints Documentation

## Base URL
```
http://YOUR_SERVER_IP:5001/api
```

---

## 1. POST `/api/pi-data` (Main Image & Sensor Data)

**Purpose:** Send image and sensor data from Raspberry Pi. Processes the image with ML (YOLO + Tesseract) and updates the fridge inventory.

**Request Body:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANS...",  // Required: Base64 encoded image
  "temperature": 3.5,                        // Optional: Temperature in Celsius
  "humidity": 65.2,                          // Optional: Humidity percentage
  "gas": 120.5,                              // Optional: Gas sensor reading (ppm)
  "event": "pi_update"                       // Optional: Event type label
}
```

**Response (Success - 200):**
```json
{
  "message": "Data processed successfully",
  "items_found": 5
}
```

**Response (Error - 400/500):**
```json
{
  "error": "No image_base64 provided."
}
```

**Use Cases:**
- Raspberry Pi sends periodic images for inventory detection
- Updates detected items in the fridge database
- Stores complete event with image, sensor data, and detected items

**Example (cURL):**
```bash
curl -X POST http://localhost:5001/api/pi-data \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgo...",
    "temperature": 3.5,
    "humidity": 65,
    "gas": 120
  }'
```

---

## 2. POST `/api/sensor-data` (Sensor Only)

**Purpose:** Send only sensor readings without processing an image. Lightweight endpoint for frequent sensor updates.

**Request Body:**
```json
{
  "temperature": 3.5,        // Sensor readings (at least one required)
  "humidity": 65.2,
  "gas": 120.5,
  "event": "sensor_reading"  // Optional: Event type label
}
```

**Response (Success - 200):**
```json
{
  "message": "Sensor data recorded successfully",
  "timestamp": "2025-01-15T10:30:45.123Z"
}
```

**Response (Error - 400):**
```json
{
  "error": "No sensor data provided."
}
```

**Use Cases:**
- Raspberry Pi sends sensor data every 30 seconds for monitoring
- Android app polls to check for alerts (temperature/gas anomalies)
- Lighter weight than full image processing
- Useful for continuous monitoring without inventory updates

**Example (cURL):**
```bash
curl -X POST http://localhost:5001/api/sensor-data \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 3.5,
    "humidity": 65.2,
    "gas": 120.5,
    "event": "sensor_reading"
  }'
```

---

## 3. GET `/api/latest-data` (Latest Image & Sensor)

**Purpose:** Retrieve the most recent image and sensor data from the fridge.

**Request Parameters:** None

**Response (Success - 200):**
```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "sensorData": {
    "temperature": 3.5,
    "humidity": 65.2,
    "gas": 120.5
  },
  "image_base64": "iVBORw0KGgoAAAANS...",
  "detected_items": [
    {
      "name": "chicken",
      "confidence": 0.95,
      "expiration_date": "2025-01-20"
    },
    {
      "name": "milk",
      "confidence": 0.92,
      "expiration_date": "2025-01-18"
    }
  ],
  "eventType": "pi_update"
}
```

**Response (Error - 404):**
```json
{
  "error": "No image data found.",
  "message": "No events with images have been recorded yet."
}
```

**Use Cases:**
- Android app fetches current fridge state
- Monitor temperature and gas levels
- Display detected items with expiration dates
- View latest camera feed

**Example (cURL):**
```bash
curl -X GET http://localhost:5001/api/latest-data
```

**Example (JavaScript/Kotlin):**
```javascript
const response = await fetch('http://localhost:5001/api/latest-data');
const data = await response.json();
console.log('Current temp:', data.sensorData.temperature);
console.log('Items:', data.detected_items);
```

---

## 4. GET `/api/fridge-contents` (Current Inventory)

**Purpose:** Get the current list of items detected in the fridge with ingredient images.

**Request Parameters:** None

**Response (Success - 200):**
```json
{
  "count": 5,
  "items": [
    {
      "name": "chicken",
      "confidence": 0.95,
      "expiration_date": "2025-01-20",
      "is_approximate": false,
      "imageUrl": "https://www.themealdb.com/images/ingredients/Chicken.png"
    },
    {
      "name": "milk",
      "confidence": 0.92,
      "expiration_date": "2025-01-18",
      "is_approximate": false,
      "imageUrl": "https://www.themealdb.com/images/ingredients/Milk.png"
    },
    {
      "name": "tomato",
      "confidence": 0.88,
      "expiration_date": null,
      "is_approximate": true,
      "imageUrl": "https://www.themealdb.com/images/ingredients/Tomato.png"
    }
  ]
}
```

**Use Cases:**
- Display inventory in Android app UI
- Show item names and expiration dates
- Display ingredient images from MealDB
- Check confidence levels of detected items

**Example (cURL):**
```bash
curl -X GET http://localhost:5001/api/fridge-contents
```

---

## 5. GET `/api/recipe` (Recipe Suggestions)

**Purpose:** Generate recipe ideas based on items expiring soon (within 7 days). Falls back to any item if nothing is expiring.

**Request Parameters:** None

**Response (Success - 200):**
```json
{
  "recipe_for_item": {
    "name": "milk",
    "image": "https://www.themealdb.com/images/ingredients/Milk.png"
  },
  "recipes": [
    {
      "title": "Chocolate Mousse",
      "description": "Mix ingredients together and chill in the fridge.",
      "thumbnail": "https://www.themealdb.com/images/media/meals/...",
      "source": "https://www.youtube.com/..."
    },
    {
      "title": "Custard Tart",
      "description": "Prepare the pastry base and fill with custard mixture.",
      "thumbnail": "https://www.themealdb.com/images/media/meals/...",
      "source": "https://www.themealdb.com"
    },
    {
      "title": "Crème Brûlée",
      "description": "Heat the milk and prepare the custard base.",
      "thumbnail": "https://www.themealdb.com/images/media/meals/...",
      "source": "https://www.youtube.com/..."
    }
  ]
}
```

**Response (No Items - 200):**
```json
{
  "recipe_for_item": {
    "name": "No items in fridge",
    "image": "https://placehold.co/200x200/404040/ffffff?text=No+Items"
  },
  "recipes": [
    {
      "title": "No items in fridge.",
      "description": "Add items to your fridge to get recipes.",
      "thumbnail": "https://placehold.co/200x200/404040/ffffff?text=No+Items",
      "source": null
    }
  ]
}
```

**Use Cases:**
- Android app displays recipe suggestions
- Prioritizes items expiring in the next 7 days
- Falls back to any item if nothing is expiring
- Integrates with TheMealDB API for real recipes
- Helps reduce food waste

**Example (cURL):**
```bash
curl -X GET http://localhost:5001/api/recipe
```

---

## API Flow Diagram

```
┌─────────────────┐
│  Raspberry Pi   │
└────────┬────────┘
         │
         ├─── POST /api/pi-data (with image + sensors)
         │         │
         │         ├─→ Python ML (YOLO + Tesseract)
         │         └─→ MongoDB (events + items)
         │
         └─── POST /api/sensor-data (sensors only, every 30s)
                   └─→ MongoDB (events)

┌─────────────────┐
│   Android App   │
└────────┬────────┘
         │
         ├─── GET /api/latest-data (check alerts)
         │         └─→ Check temp/gas thresholds
         │
         ├─── GET /api/fridge-contents (show inventory)
         │         └─→ Display items with images
         │
         └─── GET /api/recipe (get suggestions)
                   └─→ TheMealDB API
                   └─→ Display recipes
```

---

## Quick Start Examples

### For Raspberry Pi:

**Send image + sensors periodically:**
```javascript
// Every 5 minutes
setInterval(async () => {
  await fetch('http://server:5001/api/pi-data', {
    method: 'POST',
    body: JSON.stringify({
      image_base64: captureImage(),
      temperature: readTemperature(),
      humidity: readHumidity(),
      gas: readGasLevel()
    })
  });
}, 300000);

// Every 30 seconds (lightweight)
setInterval(async () => {
  await fetch('http://server:5001/api/sensor-data', {
    method: 'POST',
    body: JSON.stringify({
      temperature: readTemperature(),
      humidity: readHumidity(),
      gas: readGasLevel()
    })
  });
}, 30000);
```

### For Android App:

**Fetch current inventory:**
```kotlin
val response = apiClient.getFridgeContents()
response.items.forEach { item ->
  println("${item.name} - Expires: ${item.expiration_date}")
}
```

**Get recipe suggestions:**
```kotlin
val recipeResponse = apiClient.getRecipes()
recipeResponse.recipes.forEach { recipe ->
  println("${recipe.title}: ${recipe.description}")
}
```

**Monitor sensors for alerts:**
```kotlin
val latestData = apiClient.getLatestData()
if (latestData.sensorData.temperature > 4.0) {
  showTemperatureAlert()
}
if (latestData.sensorData.gas > 150) {
  showGasAlert()
}
```