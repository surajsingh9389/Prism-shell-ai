import { Route, Routes } from 'react-router'
import './index.css'
import Home from './pages/Home'
import QueryResponse from './pages/QueryResponse'
import { QueryProvider } from './context/queryContext'

function App() {
  return (
   <>
    <QueryProvider>
    <Routes>
      <Route path='/' element={<Home />} />
      <Route path='/response' element={<QueryResponse />} />
    </Routes>
    </QueryProvider>
   </>
  )
}

export default App
