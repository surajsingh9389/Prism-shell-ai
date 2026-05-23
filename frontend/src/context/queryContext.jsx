import { createContext, useContext, useEffect, useState } from 'react';

// Create the Context
const QueryContext = createContext(null);

// Create the Provider Component
export function QueryProvider({ children }) {
  const [responseData, setResponseData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState("");

  useEffect(() => {
    // Check if a session already exists to preserve conversational state
    let currentSessionId = localStorage.getItem("chat_session_id");
    
    if (!currentSessionId) {
      // Generate a standard RFC4122 version 4 UUID natively
      currentSessionId = crypto.randomUUID();
      localStorage.setItem("chat_session_id", currentSessionId);
    }
    
    setSessionId(currentSessionId);
  }, []);


  // Helper function to explicitly clear/reset the thread if needed
  const resetSession = () => {
    const newSessionId = crypto.randomUUID();
    localStorage.setItem("chat_session_id", newSessionId);
    setSessionId(newSessionId);
    setResponseData(null);
    setQuery("");
  };

  return (
    <QueryContext.Provider value={{ 
      responseData, 
      setResponseData, 
      isLoading, 
      setIsLoading, 
      query, 
      setQuery,
      sessionId,
      resetSession
    }}>
      {children}
    </QueryContext.Provider>
  );
}

// Custom hook for easy consumption
export function useQueryData() {
  const context = useContext(QueryContext);
  if (!context) {
    throw new Error('useQueryData must be used within a QueryProvider');
  }
  return context;
}