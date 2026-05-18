import { createContext, useContext, useState } from 'react';

// Create the Context
const QueryContext = createContext(null);

// Create the Provider Component
export function QueryProvider({ children }) {
  const [responseData, setResponseData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [query, setQuery] = useState("");

  return (
    <QueryContext.Provider value={{ responseData, setResponseData, isLoading, setIsLoading, query, setQuery }}>
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