import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)
  //<div style={{width: 417, height: 201, left: 23, top: 16, position: 'absolute', background: 'black'}} />
  return (
    <>
      <div className='top-section'>
        <div>
          <img style={{width: '100%', height: '100%', borderRadius: 4}} src="https://via.placeholder.com/459x492" />
          <h1>Avatar</h1>
        </div>
      </div>
      
      <div className='bottom-section'>
        <section className='left-section'>

        </section>

        <section className='middle-section'>

        </section>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
