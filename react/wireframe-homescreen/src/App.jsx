import React from 'react'
import { useRoutes } from 'react-router-dom'
import { Link } from 'react-router-dom'
import Donation from './pages/donation'
import ManualControl from './pages/manualcontrol'
import Piloting from './pages/pilot'
import Home from './pages/home'
import './App.css'

function App() {

  //set up routes
  let routes = useRoutes([
    {
      path: '/',
      element: <Home />
    },
    {
      path: '/manualcontrol',
      element: <ManualControl />
    },
    {
      path: '/donation',
      element: <Donation />
    },
    {
      path: '/pilot',
      element: <Piloting />
    }

  ])


  return (
    <div className='App'>
      {routes}
    </div>
  )
}

export default App
