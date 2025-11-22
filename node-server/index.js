import express from "express";
import { MongoClient, ServerApiVersion } from "mongodb";
import { Buffer } from "buffer";
import admin from "firebase-admin";

import serviceAccount from "./projet-iot-d2cac-firebase-adminsdk-fbsvc-c04a67ed07.json" assert { type: "json" };

// --- Firebase Initialization ---
function initializeFirebase() {
    try {
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount),
        });
        console.log("✅ Firebase Admin SDK initialized.");
    } catch (error) {
        console.error(
            "❌ Failed to initialize Firebase Admin SDK. Check service account path and file.",
            error
        );
        // Continue running, but without push notification capability
    }
}
initializeFirebase();

// --- Configuration ---
const app = express();
const PORT = 5001;

// Get environment variables from Docker Compose
const MONGODB_URI =
    process.env.MONGODB_URI || "mongodb://mongo-db:27017/fridgeDB";
const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://python-api:8000";

// --- New MealDB API ---
const MEALDB_API_URL = "https://www.themealdb.com/api/json/v1/1";

// Middleware to parse large JSON payloads (for base64 images)
app.use(express.json({ limit: "50mb" }));

// --- Database Connection ---
let db;
async function connectToDB() {
    try {
        const client = new MongoClient(MONGODB_URI, {
            serverApi: {
                version: ServerApiVersion.v1,
                strict: true,
                deprecationErrors: true,
            },
        });
        await client.connect();
        db = client.db(); // DB name is specified in the URI
        console.log("✅ Connected to MongoDB.");
    } catch (err) {
        console.error("❌ Failed to connect to MongoDB", err);
        process.exit(1); // Exit if we can't connect
    }
}

// --- Helper Functions ---

/**
 * Calls the Python ML service to run YOLO and Tesseract.
 */
async function invokeLocalVision(base64Image, sensorData = {}) {
    try {
        const response = await fetch(`${PYTHON_API_URL}/process_image`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_base64: base64Image,
                sensor_data: sensorData, // Pass sensor data to Python
            }),
        });

        if (!response.ok) {
            throw new Error(
                `Python API returned ${
                    response.status
                }: ${await response.text()}`
            );
        }
        return await response.json();
    } catch (error) {
        console.error("❌ Error calling Python ML service:", error.message);
        return null;
    }
}

/**
 * NEW: Calls TheMealDB API to get recipe ideas.
 * @param {string} item - The name of the ingredient to search for.
 */
async function getRecipesFromMealDB(item) {
    try {
        // 1. Search for recipes by the item
        const searchUrl = `${MEALDB_API_URL}/filter.php?i=${encodeURIComponent(
            item
        )}`;
        const searchResponse = await fetch(searchUrl);
        const searchData = await searchResponse.json();

        if (!searchData.meals || searchData.meals.length === 0) {
            return [
                {
                    title: `No recipes found for "${item}".`,
                    description: "Try a different ingredient.",
                    thumbnail:
                        "https://placehold.co/200x200/404040/ffffff?text=No+Recipes",
                    source: null,
                },
            ];
        }

        // 2. Get details for the top 3 recipes
        const topMeals = searchData.meals.slice(0, 3);
        const recipePromises = topMeals.map(async (meal) => {
            const detailUrl = `${MEALDB_API_URL}/lookup.php?i=${meal.idMeal}`;
            const detailResponse = await fetch(detailUrl);
            const detailData = await detailResponse.json();
            const recipe = detailData.meals[0];

            // Format the recipe for the user
            return {
                title: recipe.strMeal,
                // Get the first sentence of the instructions
                description: recipe.strInstructions.split(". ")[0] + ".",
                thumbnail: recipe.strMealThumb,
                source: recipe.strSource || recipe.strYoutube,
            };
        });

        const recipes = await Promise.all(recipePromises);
        return recipes;
    } catch (error) {
        console.error("❌ Error calling TheMealDB API:", error.message);
        return [
            {
                title: "Error fetching recipes.",
                description: "The recipe API might be down.",
                thumbnail: "https://placehold.co/200x200/f00/ffffff?text=Error",
                source: null,
            },
        ];
    }
}

/**
 * NEW: Formats an item name to match TheMealDB ingredient image URL.
 * @param {string} itemName - The name of the ingredient (e.g., "chicken breast")
 * @returns {string} The URL to the ingredient image.
 */
function getIngredientImageUrl(itemName) {
    if (!itemName)
        return "https://placehold.co/100x100/404040/ffffff?text=Unknown";

    // (e.g., "chicken breast" -> "Chicken%20Breast")
    const formattedItemName = itemName
        .split(" ")
        .map(
            (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
        )
        .join("%20");

    return `https://www.themealdb.com/images/ingredients/${formattedItemName}.png`;
}

/**
 * Sends a push notification to a list of FCM tokens.
 * @param {Object} message - The full notification message object (including notification, data, and tokens fields).
 */
async function sendPushNotification(message) {
    const tokens = message.tokens;
    if (!tokens || tokens.length === 0) {
        console.log("No subscribed devices to send notification to.");
        return;
    }

    try {
        // Use sendMulticast for sending to multiple devices at once
        const response = await admin.messaging().sendEachForMulticast(message);
        console.log(
            "✅ Successfully sent push notification:",
            response.successCount,
            "sent,",
            response.failureCount,
            "failed."
        );
        if (response.failureCount > 0) {
            // Handle failed tokens (e.g., remove from DB if they are invalid)
            response.responses.forEach((res, index) => {
                if (!res.success) {
                    console.warn(
                        `Failed token: ${tokens[index]} - Error: ${res.error.code}`
                    );
                    // TODO: Implement logic to delete invalid tokens from 'subscriptions' collection
                }
            });
        }
    } catch (error) {
        console.error("❌ Error sending push notification:", error);
    }
}

// --- API Endpoints ---

/**
 * [POST /api/pi-data]
 * Main endpoint for the Raspberry Pi.
 * Receives image and sensor data, processes it, and updates the database.
 */
app.post("/api/pi-data", async (req, res) => {
    const { image_base64, event, ...sensorData } = req.body;

    if (!image_base64) {
        return res.status(400).json({ error: "No image_base64 provided." });
    }

    // 1. Call Python ML service for vision processing
    console.log("Receiving image, sending to Python ML service...");
    // Pass sensor data (temp, humidity, gas) to the vision service
    const visionData = await invokeLocalVision(image_base64, sensorData);

    if (!visionData) {
        return res
            .status(500)
            .json({ error: "Failed to process image with ML service." });
    }

    console.log("✅ Vision data received:", visionData.detected_items);

    const newEvent = {
        eventType: event || "pi_update",
        timestamp: new Date(),
        sensorData: sensorData,
        image_base64: image_base64, // Store the image
        ...visionData,
    };

    try {
        // 2. Log the full event to the 'events' collection
        const eventsCollection = db.collection("events");
        await eventsCollection.insertOne(newEvent);

        // 3. Update the 'fridge_items' collection to reflect the current state
        const itemsCollection = db.collection("fridge_items");
        await itemsCollection.deleteMany({}); // Clear the current state
        if (visionData.detected_items && visionData.detected_items.length > 0) {
            // @ts-ignore
            await itemsCollection.insertMany(visionData.detected_items); // Insert new state
        }

        console.log("✅ Database updated.");
        res.status(200).json({
            message: "Data processed successfully",
            items_found: visionData.detected_items.length,
        });
    } catch (err) {
        console.error("❌ Error updating database:", err);
        res.status(500).json({ error: "Failed to save data to database." });
    }
});

/**
 * [POST /api/sensor-data]
 * Endpoint for the Pi to send only sensor data (without image).
 * Stores sensor readings for monitoring AND checks for alert conditions.
 */
app.post("/api/sensor-data", async (req, res) => {
    const { event, ...sensorData } = req.body;

    if (!sensorData || Object.keys(sensorData).length === 0) {
        return res.status(400).json({ error: "No sensor data provided." });
    }

    // Ensure temperature and gas are treated as numbers for comparison
    const temp = parseFloat(sensorData.temperature);
    const gas = parseFloat(sensorData.gas);

    const sensorEvent = {
        eventType: event || "sensor_reading",
        timestamp: new Date(),
        sensorData: sensorData,
    };

    try {
        const eventsCollection = db.collection("events");
        await eventsCollection.insertOne(sensorEvent);

        console.log("✅ Sensor data logged:", sensorData);

        // --- NEW: ALERT CONDITION CHECK ---
        const isTempAlert = temp > 25;
        const isGasAlert = gas > 50;

        if (isTempAlert || isGasAlert) {
            console.log(
                "🚨 CRITICAL ALERT TRIGGERED: High Temp or Gas Detected!"
            );

            // 1. Fetch all subscribed device tokens
            const subscriptionsCollection = db.collection("subscriptions");
            const deviceTokens = (
                await subscriptionsCollection
                    .find({}, { projection: { fcmToken: 1, _id: 0 } })
                    .toArray()
            )
                .map((sub) => sub.fcmToken)
                .filter((token) => token);

            // 2. Prepare the alert message
            if (deviceTokens.length > 0) {
                const alertData = {
                    temperature: temp,
                    humidity: sensorData.humidity,
                    gas: gas,
                    alert_type:
                        isTempAlert && isGasAlert
                            ? "BOTH"
                            : isTempAlert
                            ? "TEMP"
                            : "GAS",
                };

                let alertTitle = "⚠️ Fridge CRITICAL ALERT!";
                let alertBody = "";

                if (isTempAlert) {
                    alertBody += `Temperature is high: ${temp}°C. `;
                }
                if (isGasAlert) {
                    alertBody += `Gas concentration is high: ${gas}.`;
                }

                const notificationData = {
                    notification: {
                        title: alertTitle,
                        body: alertBody.trim(),
                        // Add icon/color details here if needed for Android
                    },
                    data: {
                        ...alertData,
                        timestamp: sensorEvent.timestamp.toISOString(),
                        type: "alert_critical",
                        // Ensure all data payload values are strings for FCM compatibility
                        temperature: String(temp),
                        gas: String(gas),
                        humidity: String(sensorData.humidity),
                    },
                    tokens: deviceTokens,
                };

                // 3. Send the push notification
                await sendPushNotification(notificationData);
            }
        }
        // --- END ALERT CHECK ---

        res.status(200).json({
            message: "Sensor data recorded successfully",
            timestamp: sensorEvent.timestamp,
        });
    } catch (err) {
        console.error("❌ Error logging sensor data/sending alert:", err);
        res.status(500).json({
            error: "Failed to save sensor data to database or send alert.",
        });
    }
});

/**
 * [GET /api/latest-image]
 * Returns only the latest image and its timestamp.
 */
app.get("/api/latest-image", async (req, res) => {
    try {
        const eventsCollection = db.collection("events");

        // Find the latest event with an image
        const latestEventWithImage = await eventsCollection.findOne(
            { image_base64: { $exists: true, $ne: null } },
            { sort: { timestamp: -1 } }
        );

        if (!latestEventWithImage) {
            return res.status(404).json({
                error: "No image data found.",
                message: "No events with images have been recorded yet.",
            });
        }

        res.status(200).json({
            timestamp: latestEventWithImage.timestamp,
            image_base64: latestEventWithImage.image_base64,
            eventType: latestEventWithImage.eventType,
        });
    } catch (err) {
        console.error("❌ Error fetching latest image:", err);
        res.status(500).json({ error: "Failed to retrieve latest image." });
    }
});

/**
 * [POST /api/subscribe]
 * Receives an FCM registration token from an Android device and stores it
 * for push notifications.
 * Request Body: { "fcmToken": "your_unique_device_token" }
 */
app.post("/api/subscribe", async (req, res) => {
    const { fcmToken } = req.body;

    if (!fcmToken) {
        return res
            .status(400)
            .json({ error: "Missing fcmToken in request body." });
    }

    try {
        // Ensure the db connection is ready
        if (!db) {
            await connectToDB();
        }

        const subscriptionsCollection = db.collection("subscriptions");

        // Use the token as the unique ID for upserting (insert or update)
        const result = await subscriptionsCollection.updateOne(
            { _id: fcmToken }, // Filter by token
            {
                $set: {
                    fcmToken: fcmToken,
                    platform: "android", // Optional: useful for targeting
                    subscribedAt: new Date(),
                },
            },
            { upsert: true } // If the token doesn't exist, insert it
        );

        let message;
        if (result.upsertedCount > 0) {
            message = "Device subscribed successfully.";
        } else {
            message = "Device token already subscribed (updated timestamp).";
        }

        console.log(
            `✅ Subscription successful for token: ${fcmToken.substring(
                0,
                15
            )}...`
        );
        res.status(200).json({ message: message });
    } catch (err) {
        console.error("❌ Error subscribing device:", err);
        res.status(500).json({
            error: "Failed to subscribe device due to a server error.",
        });
    }
});

/**
 * [GET /api/latest-sensor-data]
 * Returns only the latest sensor data and its timestamp.
 */
app.get("/api/latest-sensor-data", async (req, res) => {
    try {
        const eventsCollection = db.collection("events");

        // Find the latest event with sensor data
        const latestEventWithSensor = await eventsCollection.findOne(
            { sensorData: { $exists: true, $ne: null } },
            { sort: { timestamp: -1 } }
        );

        if (!latestEventWithSensor) {
            return res.status(404).json({
                error: "No sensor data found.",
                message: "No sensor readings have been recorded yet.",
            });
        }

        res.status(200).json({
            timestamp: latestEventWithSensor.timestamp,
            sensorData: latestEventWithSensor.sensorData,
            eventType: latestEventWithSensor.eventType,
        });
    } catch (err) {
        console.error("❌ Error fetching latest sensor data:", err);
        res.status(500).json({
            error: "Failed to retrieve latest sensor data.",
        });
    }
});

/**
 * [GET /api/latest-data]
 * UPDATED: Returns both latest image and sensor data together (kept for backwards compatibility).
 */
app.get("/api/latest-data", async (req, res) => {
    try {
        const eventsCollection = db.collection("events");

        // Find the latest event with an image
        const latestEventWithImage = await eventsCollection.findOne(
            { image_base64: { $exists: true, $ne: null } },
            { sort: { timestamp: -1 } }
        );

        // Find the latest event with sensor data
        const latestEventWithSensor = await eventsCollection.findOne(
            { sensorData: { $exists: true, $ne: null } },
            { sort: { timestamp: -1 } }
        );

        if (!latestEventWithImage && !latestEventWithSensor) {
            return res.status(404).json({
                error: "No data found.",
                message: "No events have been recorded yet.",
            });
        }

        const response = {
            image_base64: latestEventWithImage
                ? latestEventWithImage.image_base64
                : null,
            imageTimestamp: latestEventWithImage
                ? latestEventWithImage.timestamp
                : null,
            sensorTimestamp: latestEventWithSensor
                ? latestEventWithSensor.timestamp
                : null,
            sensorData: latestEventWithSensor
                ? latestEventWithSensor.sensorData
                : null,
            sensorEventType: latestEventWithSensor
                ? latestEventWithSensor.eventType
                : null,
        };

        res.status(200).json(response);
    } catch (err) {
        console.error("❌ Error fetching latest data:", err);
        res.status(500).json({ error: "Failed to retrieve latest data." });
    }
});

/**
 * [GET /api/fridge-contents]
 * Returns the current items detected in the fridge.
 */
app.get("/api/fridge-contents", async (req, res) => {
    try {
        const itemsFromDB = await db
            .collection("fridge_items")
            .find({})
            .toArray();

        // --- MODIFIED: Add image URL to each item ---
        const items = itemsFromDB.map((item) => {
            return {
                ...item, // Spread the original item properties
                imageUrl: getIngredientImageUrl(item.name), // Add the new image URL
            };
        });

        res.status(200).json({
            count: items.length,
            items: items, // Send the modified items array
        });
    } catch (err) {
        console.error("❌ Error fetching fridge contents:", err);
        res.status(500).json({ error: "Failed to retrieve data." });
    }
});

/**
 * [GET /api/recipe]
 * Generates recipe ideas based on items expiring soon, using TheMealDB.
 */
app.get("/api/recipe", async (req, res) => {
    try {
        const now = new Date();
        const oneWeekFromNow = new Date(
            now.getTime() + 7 * 24 * 60 * 60 * 1000
        ); // Correct way to add 7 days

        // Find items that have an expiration date (and are not approx)
        // and are expiring within 7 days
        const expiringItems = await db
            .collection("fridge_items")
            .find({
                expiration_date: {
                    $ne: null,
                    $lte: oneWeekFromNow.toISOString(),
                },
                is_approximate: false, // Only use items with a real, scanned date
            })
            .toArray();

        let itemToSearch = null;

        if (expiringItems.length > 0) {
            // Use the item expiring the earliest
            expiringItems.sort((a, b) => {
                return (
                    new Date(a.expiration_date).getTime() -
                    new Date(b.expiration_date).getTime()
                );
            });
            itemToSearch = expiringItems[0].name;
        } else {
            // If nothing is expiring, just use any 1 item
            const anyItem = await db.collection("fridge_items").findOne();
            if (anyItem) {
                itemToSearch = anyItem.name;
            }
        }

        if (!itemToSearch) {
            return res.status(200).json({
                recipe_for_item: {
                    name: "No items in fridge",
                    image: "https://placehold.co/200x200/404040/ffffff?text=No+Items",
                },
                recipes: [
                    {
                        title: "No items in fridge.",
                        description: "Add items to your fridge to get recipes.",
                        thumbnail:
                            "https://placehold.co/200x200/404040/ffffff?text=No+Items",
                        source: null,
                    },
                ],
            });
        }

        console.log(`Generating recipes for: ${itemToSearch}`);

        // --- Format item name for TheMealDB image URL ---
        const itemImageUrl = getIngredientImageUrl(itemToSearch);

        const recipes = await getRecipesFromMealDB(itemToSearch);

        // --- Modify the response to include the item details ---
        res.status(200).json({
            recipe_for_item: {
                name: itemToSearch,
                image: itemImageUrl,
            },
            recipes: recipes,
        });
    } catch (err) {
        console.error("❌ Error generating recipe:", err);
        res.status(500).json({ error: "Failed to generate recipe." });
    }
});

// --- Start Server ---
app.listen(PORT, "0.0.0.0", async () => {
    await connectToDB();
    console.log(`✅ Node.js server listening on http://0.0.0.0:${PORT}`);
});
