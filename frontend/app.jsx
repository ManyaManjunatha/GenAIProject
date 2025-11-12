import React, { useState } from "react";
import { analyzeText } from "./api";
import SubmissionForm from "./components/SubmissionForm";
import Dashboard from "./components/Dashboard";

const App = () => {
  const [result, setResult] = useState(null);

  const handleAnalyze = async (text) => {
    const data = await analyzeText(text);
    setResult(data);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-8">
      <h1 className="text-3xl font-bold mb-6 text-green-700">
        AI Misinformation Detector
      </h1>
      <SubmissionForm onAnalyze={handleAnalyze} />
      {result && <Dashboard result={result} />}
    </div>
  );
};

export default App;
