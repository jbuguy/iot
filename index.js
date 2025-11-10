import express from 'express';
import { appendFile } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { MongoClient, ServerApiVersion } from 'mongodb';
import { HfInference } from '@huggingface/inference';
import { spawn } from 'child_process';

// --- Configuration ---
const app = express();
const PORT = 5001;

// ES module equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const LOG_FILE = path.join(__dirname, 'pi_data_log.txt');

// --- Hugging Face API Configuration ---
const HF_API_KEY = process.env.HF_TOKEN; // <-- PASTE YOUR HF API KEY HERE
const hf = new HfInference(HF_API_KEY);


// --- MongoDB Configuration ---
const MONGO_URL = process.env.MONGO_URL || 'mongodb://localhost:27017';
const DB_NAME = 'pi_database';
const EVENTS_COLLECTION_NAME = 'events';
const ITEMS_COLLECTION_NAME = 'fridge_items'; // <-- New collection for fridge state

// --- Database Connection ---
let eventsCollection;
let itemsCollection; // <-- New collection variable
const mongoClient = new MongoClient(MONGO_URL);

async function connectToMongo() {
  try {
    await mongoClient.connect();
    console.log(`✅ Connected to MongoDB at ${MONGO_URL}`);
    const db = mongoClient.db(DB_NAME);
    eventsCollection = db.collection(EVENTS_COLLECTION_NAME);
    itemsCollection = db.collection(ITEMS_COLLECTION_NAME); // <-- Initialize new collection
  } catch (err) {
    console.error("Fatal: Error connecting to MongoDB:", err.message);
    process.exit(1);
  }
}

// --- Middleware ---
app.use(express.json({ limit: '50mb' }));

// --- Sensor Logic Function ---
function runSensorLogic(data) {
  // ... (Same as before, no changes needed)
  const { gas_level_percent, door_status } = data;
  let is_alert = false;
  let alert_reason = "Sensor: All clear.";
  if (gas_level_percent > 50) {
    is_alert = true;
    alert_reason = "Sensor Alert: High gas level detected!";
  } else if (door_status === 'open') {
    is_alert = true;
    alert_reason = "Sensor Alert: Door is open.";
  }
  return { is_alert, alert_reason };
}

// --- Local Vision Script Function ---
/**
 * Invokes the local Python vision script (YOLO + OCR).
 * @param {object} data - The JSON data received from the Pi (must contain image_base64).
 * @returns {Promise<object>} - A promise that resolves with the JSON output from the script.
 */
function invokeLocalVision(data) {
  return new Promise((resolve, reject) => {
    console.log("Spawning local vision process (YOLO + OCR)...");
    // Use 'python3'. Ensure 'local_vision.py' is in the same directory.
    const pythonProcess = spawn('python3', [path.join(__dirname, 'local_vision.py')]);

    let scriptOutput = "";
    let scriptError = "";

    pythonProcess.stdout.on('data', (data) => {
      scriptOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      scriptError += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Local vision script error (Code ${code}): ${scriptError}`);
        return reject(new Error(`Python script failed: ${scriptError}`));
      }
      try {
        const result = JSON.parse(scriptOutput);
        if (result.error) {
          return reject(new Error(result.error));
        }
        console.log("Local vision process finished.");
        resolve(result); // e.g., { detected_items: [...] }
      } catch (e) {
        console.error("Failed to parse Python script output:", scriptOutput);
        reject(new Error('Failed to parse Python output.'));
      }
    });

    try {
      pythonProcess.stdin.write(JSON.stringify(data));
      pythonProcess.stdin.end();
    } catch (e) {
      reject(new Error(`Failed to write to python process: ${e.message}`));
    }
  });
}

// --- Hugging Face Recipe Function ---
/**
 * Calls HF API for recipe generation ONLY.
 * @param {string[]} objects - The list of objects to make a recipe with.
 */
async function getRecipeFromHF(objects) {
  let recipeResult = {
    // Change from single object to an array
    recipes: []
  };

  if (objects && objects.length > 0) {
    try {
      console.log("Calling HF Text Generation (for Recipe)...");
      // Updated prompt to ask for 3 recipe ideas in a parsable format
      const prompt = `Create a list of 3 simple recipe ideas using: ${objects.join(', ')}. Prioritize using these items first. For each recipe, provide a title and a one-sentence description. Format each as: "Title: [Recipe Title]\nDescription: [Recipe Description]" and separate each entry with "---".`;
      
      const recipeTextResponse = await hf.textGeneration({
        model: 'google/gemma-2b-it',
        inputs: prompt,
        // Increased tokens to allow for a list
        parameters: { max_new_tokens: 400, do_sample: true }
      });
      
      const generatedText = recipeTextResponse.generated_text.replace(prompt, '').trim();
      
      // New parsing logic to handle multiple recipes
      const recipeBlocks = generatedText.split('---'); // Split by our separator
      const parsedRecipes = [];
      
      const titleRegex = /Title:\s*(.*)/i; // Case-insensitive
      const descRegex = /Description:\s*(.*)/i; // Case-insensitive

      for (const block of recipeBlocks) {
        const titleMatch = block.match(titleRegex);
        const descMatch = block.match(descRegex);

        if (titleMatch && descMatch) {
          parsedRecipes.push({
            title: titleMatch[1].trim(),
            description: descMatch[1].trim()
          });
        } else if (block.trim().length > 10) { 
          // Fallback for models that don't follow instructions perfectly
          const lines = block.trim().split('\n');
          if (lines.length >= 2) {
             parsedRecipes.push({
                title: lines[0].replace(/^[0-9]\.\s*/, '').trim(), // Remove "1." etc.
                description: lines[1].trim()
             });
          }
        }
      }

      recipeResult.recipes = parsedRecipes;
      
      // Handle case where parsing failed but we got text
      if (recipeResult.recipes.length === 0 && generatedText.length > 0) {
          recipeResult.recipes = [{
              title: "Generated Recipes (Parsing Failed)",
              description: generatedText // Put the whole text as description
          }];
      }

    } catch (err) {
      console.error(`HF Text Generation Error: ${err.message}`);
      // Update error state to match new data structure
      recipeResult.recipes = [{
         title: "Recipe Generation Failed",
         description: "Could not generate recipe. Check server logs."
      }];
    }
  }
  return recipeResult;
}


// --- Routes ---
app.get('/', (req, res) => {
  res.send('Node.js Hybrid ML server for Raspberry Pi is running. Send POST data to /api/pi-data.');
});

// --- MODIFIED: Main Pi Data Route ---
app.post('/api/pi-data', async (req, res) => {
  try {
    // 1. Get data from Pi
    const data = req.body;
    if (!data || !data.image_base64) {
      console.warn("Received request with no image data.");
      return res.status(400).json({ status: "error", message: "No image_base64 data received" });
    }
    const server_timestamp = new Date();
    data.server_timestamp = server_timestamp.toISOString();

    // 2. Log to console
    console.log(`[${data.server_timestamp}] --- Data Received (Event: ${data.event || 'N/A'}) ---`);

    // 3. Log to file
    appendFile(LOG_FILE, JSON.stringify({ event: data.event, timestamp: data.server_timestamp }) + '\n')
      .catch(err => console.error("Failed to write to log file:", err));

    // 4. Invoke Local Vision (YOLO + OCR)
    const visionResult = await invokeLocalVision(data);
    console.log("Local Vision Result:", visionResult);

    // 5. Run sensor logic
    const sensorResult = runSensorLogic(data);
    
    // 6. Combine ML results
    const finalMLResult = {
      detected_items: visionResult.detected_items || [],
      is_alert: sensorResult.is_alert,
      alert_reason: sensorResult.alert_reason,
      // Note: We'll generate the recipe "on-demand" via the new API
    };
    
    console.log("--- Combined ML Result ---");
    console.log(JSON.stringify(finalMLResult, null, 2));

    // 7. Update Fridge State in MongoDB
    try {
      // Clear the old fridge state
      await itemsCollection.deleteMany({});
      
      // If we detected items, insert them
      if (finalMLResult.detected_items.length > 0) {
        const itemsToInsert = finalMLResult.detected_items.map(item => ({
          ...item,
          // Convert string date to JS Date object for better querying
          expiration_date: item.expiration_date ? new Date(item.expiration_date) : null,
          last_seen: server_timestamp
        }));
        await itemsCollection.insertMany(itemsToInsert);
        console.log(`Updated fridge_items collection with ${itemsToInsert.length} items.`);
      } else {
        console.log("No items detected, fridge_items collection is now empty.");
      }
    } catch (dbErr) {
      console.error("Failed to update fridge_items collection:", dbErr.message);
    }

    // 8. Save full event to log collection
    try {
      const documentToInsert = {
        ...data,
        ml_result: finalMLResult
      };
      eventsCollection.insertOne(documentToInsert)
        .then(result => {
          console.log(`New event (ID: ${result.insertedId}) saved to events collection.`);
        });
    } catch (e) {
      console.error("Event log insert operation error:", e.message);
    }

    // 9. Send 200 OK back to Pi
    res.status(200).json({
      status: "success",
      received_event: data.event,
      ml_result: finalMLResult
    });

  } catch (error) {
    console.error(`[ERROR] Failed to process request: ${error.message}`);
    res.status(500).json({ status: "error", message: error.message, details: error.stack });
  }
});

// --- NEW API ENDPOINT: Get Fridge Contents ---
app.get('/api/fridge-contents', async (req, res) => {
  try {
    const items = await itemsCollection.find({}).toArray();
    res.status(200).json({
      status: "success",
      count: items.length,
      items: items
    });
  } catch (error) {
    console.error("Failed to get fridge contents:", error.message);
    res.status(500).json({ status: "error", message: "Failed to retrieve data from database." });
  }
});

// --- NEW API ENDPOINT: Get Recipe ---
app.get('/api/recipe', async (req, res) => {
  try {
    const days_to_expire = req.query.days ? parseInt(req.query.days, 10) : 3;
    
    const today = new Date();
    const expiry_limit = new Date();
    expiry_limit.setDate(today.getDate() + days_to_expire);

    console.log(`Getting recipe for items expiring between ${today.toISOString()} and ${expiry_limit.toISOString()}`);

    // Find items that have an expiration date
    // between today and the limit
    const expiringItems = await itemsCollection.find({
      expiration_date: {
        $gte: today,
        $lte: expiry_limit
      }
    }).toArray();

    let itemsList = expiringItems.map(item => item.name);

    if (itemsList.length === 0) {
      // If no items are expiring, just grab 3 random items
      console.log("No expiring items, grabbing random items...");
      const randomItems = await itemsCollection.aggregate([
        { $sample: { size: 3 } }
      ]).toArray();
      itemsList = randomItems.map(item => item.name);
    }

    if (itemsList.length === 0) {
      return res.status(404).json({
        status: "error",
        message: "No items found in fridge to make a recipe."
      });
    }

    // Call HF to get a recipe
    const recipe = await getRecipeFromHF(itemsList);
    
    res.status(200).json({
      status: "success",
      recipe_for: itemsList,
      recipe: recipe
    });

  } catch (error) {
    console.error("Failed to generate recipe:", error.message);
    res.status(500).json({ status: "error", message: "Failed to generate recipe." });
  }
});


// --- Start Server ---
console.log("Connecting to MongoDB...");
connectToMongo().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Node.js server listening on http://0.0.0.0:${PORT}`);
    console.log(`Logging data to ${LOG_FILE}`);
  });
}).catch(err => {
  console.error("Failed to start server:", err);
});