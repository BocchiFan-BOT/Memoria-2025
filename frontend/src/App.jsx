import { useState } from "react";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard";

function App() {
  const [logged, setLogged] = useState(false);

  return (
    <div>
      {logged ? <Dashboard /> : <Login onSuccess={() => setLogged(true)} />}
    </div>
  );
}

export default App;


