import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import SpamClassifier from './SpamClassifier (1).jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>    
    <SpamClassifier />
  </StrictMode>,
)
