const express = require("express");
const cors = require("cors");
const axios=require("axios");

const app = express();
app.use(express.json());
app.use(cors())

app.post("/api/model", async (req, res) => {

    try {
        const {email}=req.body ;

        const pythonResponse =await axios.post("http://localhost:5001/api/predict",{
        email:email
        });

        const data = pythonResponse.data;

        res.json([
            content:            
        ])

        

return data;
    }
    catch(err){console.log} 
    
}

app.listen(port=5000);