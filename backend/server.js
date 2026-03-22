const express = require('express'); 
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json());

app.post("/api/model", async (req, res) => {
    try {
        const {email} = req.body; 

        const pythonResponse=await axios.post(process.env.PYTHON_SERVICE_URL + "/api/predict",{
            email:email
        });

        const {prediction,confidence}=pythonResponse.data; 

        res.json({
            content: [
                {
                    text: JSON.stringify({
                        classification: prediction==0 ? "LEGIT" : "SPAM",
                        confidence: confidence,
                        indicators: ["Keyword analysis", "Pattern matching"],
                        tags: ["TAG1", "TAG2"],
                        summary: "one sentence summary"
                    })
                }
            ]
        })
    }
    catch (err) { console.error("DETAILED ERROR:", err.response ? err.response.data : err.message);
        res.status(500).json({ error: "Backend bridge failed" }); }    
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}.`);
});