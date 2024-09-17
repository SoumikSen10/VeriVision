import { asyncHandler } from "../utils/asyncHandler.js";
import { spawn } from "child_process";
import path from "path";
import fs from "fs/promises";
import { fileURLToPath } from "url";

// Fix for __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Controller for handling deepfake detection
const deepfakeDetection = asyncHandler(async (req, res) => {
  try {
    // Ensure the file is uploaded
    if (!req.file) {
      return res.status(400).json({ message: "No video file uploaded" });
    }

    // Path to the uploaded video in 'uploads' folder inside 'src'
    const videoPath = path.resolve(__dirname, "../uploads", req.file.filename);

    // Check if the uploaded video exists
    try {
      await fs.access(videoPath); // Check if the file exists in the directory
    } catch (error) {
      return res.status(404).json({ message: "Uploaded video not found" });
    }

    // Construct the absolute path to the Python script in 'ML' folder
    // Correct path from 'controllers' to 'ML' (since they are siblings to 'Backend')
    const pythonScriptPath = path.resolve(
      __dirname,
      "../../../ML/extractScript.py"
    );

    console.log("Python Script Path:", pythonScriptPath);

    // Check if the Python script exists
    try {
      await fs.access(pythonScriptPath); // Ensure the Python script exists at the expected location
    } catch (error) {
      return res.status(404).json({ message: "Python script not found" });
    }

    // Spawn the Python process to run the deepfake detection, passing the video path as an argument
    const pythonProcess = spawn("python", [pythonScriptPath, videoPath]);

    let scriptOutput = "";

    // Listen for data from the Python script's stdout
    pythonProcess.stdout.on("data", (data) => {
      scriptOutput += data.toString();
    });

    // Handle Python script errors
    pythonProcess.stderr.on("data", (data) => {
      console.error("Python script error:", data.toString());
    });

    // When the Python process is done
    pythonProcess.on("close", async (code) => {
      if (code === 0) {
        res.status(200).json({
          message: "Deepfake detection completed",
          result: scriptOutput,
        });

        // Optionally remove the video after successful processing
        try {
          await fs.unlink(videoPath); // Remove the file after processing
          console.log("Uploaded video deleted successfully.");
        } catch (err) {
          console.error("Error deleting uploaded video:", err);
        }
      } else {
        res.status(500).json({ message: "Python script failed", code });
      }
    });
  } catch (error) {
    console.error("Error during deepfake detection:", error);
    res.status(500).json({ message: "Internal server error" });
  }
});

export { deepfakeDetection };
