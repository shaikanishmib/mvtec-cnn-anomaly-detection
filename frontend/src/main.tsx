import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'
performance.mark('app:start')
createRoot(document.getElementById('root')!).render(<App />)
