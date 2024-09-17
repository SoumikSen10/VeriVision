import multer from "multer";
import path from "path";

// Define storage for multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.resolve("src/uploads")); // Save files in 'uploads' folder
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`); // Save file with timestamp
  },
});

// File filter (optional)
const fileFilter = (req, file, cb) => {
  if (file.mimetype === "video/mp4" || file.mimetype === "video/mkv") {
    cb(null, true); // Accept video files only
  } else {
    cb(new Error("Invalid file type. Only MP4 or MKV allowed."), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
});

export { upload };
