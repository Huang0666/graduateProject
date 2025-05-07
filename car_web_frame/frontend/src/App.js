import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import SingleCar from './pages/SingleCar';
import CarType from './pages/CarType';
import WordCloud from './pages/WordCloud';
import './App.css';

const { Header, Content } = Layout;

function App() {
  return (
    <Router>
      <Layout className="layout" style={{ minHeight: '100vh' }}>
        <Header>
          <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['1']}>
            <Menu.Item key="1">
              <Link to="/">单车型分析</Link>
            </Menu.Item>
            <Menu.Item key="2">
              <Link to="/car-type">车型类别分析</Link>
            </Menu.Item>
            <Menu.Item key="3">
              <Link to="/word-cloud">词云分析</Link>
            </Menu.Item>
          </Menu>
        </Header>
        <Content style={{ padding: '50px' }}>
          <Routes>
            <Route path="/" element={<SingleCar />} />
            <Route path="/car-type" element={<CarType />} />
            <Route path="/word-cloud" element={<WordCloud />} />
          </Routes>
        </Content>
      </Layout>
    </Router>
  );
}

export default App;
