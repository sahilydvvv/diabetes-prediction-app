const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));

// Serve HTML from public folder
app.use(express.static(path.join(__dirname, "public")));

app.post("/predict", (req, res) => {
    const input = [
        req.body.Pregnancies,
        req.body.Glucose,
        req.body.BloodPressure,
        req.body.SkinThickness,
        req.body.Insulin,
        req.body.BMI,
        req.body.DiabetesPedigreeFunction,
        req.body.Age
    ];

    const python = spawn("python", ["predict.py", ...input]);

    python.stdout.on("data", (data) => {
        const result = parseInt(data.toString()) === 1 
            ? "Diabetic" 
            : "Not Diabetic";

        res.redirect("/?result=" + result);
    });
});

app.listen(3000, () => {
    console.log("Server running at http://localhost:3000");
});
