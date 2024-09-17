import { Router } from "express";
import { deepfakeDetection } from "../controllers/detector.controller.js";
import { verifyJWT } from "../middlewares/auth.middleware.js";
import { upload } from "../middlewares/multer.middleware.js"; // Import multer middleware

const router = Router();

// Route for deepfake detection
// Secured route: User needs to be verified via JWT to upload the video file
router
  .route("/detect")
  .post(verifyJWT, upload.single("video"), deepfakeDetection); // Added multer middleware to handle video upload

export default router;
