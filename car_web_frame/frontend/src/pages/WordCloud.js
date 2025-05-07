import React, { useState, useEffect } from 'react';
import { Card, Select, Row, Col, Empty, Spin } from 'antd';
import ReactECharts from 'echarts-for-react';
import 'echarts-wordcloud';
import axios from 'axios';

const { Option } = Select;

const sentimentTypes = [
  { value: 0, label: '负面' },
  { value: 1, label: '正面' },
  { value: 2, label: '中性' },
  { value: 3, label: '无关' },
  { value: 4, label: '全部' }
];

const carTypes = ['燃油', '新能源', 'SUV', 'MPV', '轿车'];

function WordCloud() {
  const [selectionType, setSelectionType] = useState('car'); // 'car' or 'type'
  const [cars, setCars] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [selectedSentiment, setSelectedSentiment] = useState(1); // 设置默认值为1（正向）
  const [keywords, setKeywords] = useState([]);
  const [loading, setLoading] = useState(false);

  // 获取所有车型并设置默认值
  useEffect(() => {
    axios.get('http://localhost:5000/api/cars/names')
      .then(response => {
        setCars(response.data);
        // 找到丰田卡罗拉的car_id并设置为默认值
        const carolla = response.data.find(car => car.car_name.includes('卡罗拉'));
        if (carolla) {
          setSelectedItem(carolla.car_id);
        }
      })
      .catch(error => console.error('Error fetching cars:', error));
  }, []);

  // 组件挂载时自动获取数据
  useEffect(() => {
    if (selectedItem && selectedSentiment !== null) {
      fetchKeywords();
    }
  }, []); // 空依赖数组，只在组件挂载时执行一次

  // 获取关键词
  const fetchKeywords = async () => {
    if (!selectedItem || selectedSentiment === null) return;

    setLoading(true);
    try {
      let response;
      if (selectionType === 'car') {
        response = await axios.get('http://localhost:5000/api/car/keywords', {
          params: {
            car_id: selectedItem,
            sentiment_type: selectedSentiment
          }
        });
      } else {
        response = await axios.get('http://localhost:5000/api/car-type/keywords', {
          params: {
            category: selectedItem,
            sentiment_type: selectedSentiment
          }
        });
      }
      
      // 处理关键词数据
      const keywordData = response.data.map(item => ({
        name: selectionType === 'car' ? item.keywords : item.keyword,
        value: Math.random() * 100 // 这里可以根据实际需求设置词的大小
      }));
      
      setKeywords(keywordData);
    } catch (error) {
      console.error('Error fetching keywords:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchKeywords();
  }, [selectedItem, selectedSentiment]);

  const getWordCloudOption = () => {
    return {
      tooltip: {
        show: true
      },
      series: [{
        type: 'wordCloud',
        shape: 'circle',
        left: 'center',
        top: 'center',
        width: '70%',
        height: '80%',
        right: null,
        bottom: null,
        sizeRange: [12, 60],
        rotationRange: [-90, 90],
        rotationStep: 45,
        gridSize: 8,
        drawOutOfBound: false,
        textStyle: {
          fontFamily: 'sans-serif',
          fontWeight: 'bold',
          color: function () {
            return 'rgb(' + [
              Math.round(Math.random() * 160),
              Math.round(Math.random() * 160),
              Math.round(Math.random() * 160)
            ].join(',') + ')';
          }
        },
        emphasis: {
          textStyle: {
            shadowBlur: 10,
            shadowColor: '#333'
          }
        },
        data: keywords
      }]
    };
  };

  const handleSelectionTypeChange = (value) => {
    setSelectionType(value);
    setSelectedItem(null);
    setKeywords([]);
  };

  return (
    <div>
      <Card title="词云分析">
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Select
              style={{ width: 200, marginRight: 16 }}
              placeholder="选择分析类型"
              value={selectionType}
              onChange={handleSelectionTypeChange}
            >
              <Option value="car">按车型</Option>
              <Option value="type">按类别</Option>
            </Select>

            <Select
              style={{ width: 200, marginRight: 16 }}
              placeholder={selectionType === 'car' ? '选择车型' : '选择类别'}
              value={selectedItem}
              onChange={setSelectedItem}
            >
              {selectionType === 'car' ? (
                cars.map(car => (
                  <Option key={car.car_id} value={car.car_id}>{car.car_name}</Option>
                ))
              ) : (
                carTypes.map(type => (
                  <Option key={type} value={type}>{type}</Option>
                ))
              )}
            </Select>
            
            <Select
              style={{ width: 200 }}
              placeholder="选择情感类型"
              value={selectedSentiment}
              onChange={setSelectedSentiment}
            >
              {sentimentTypes.map(type => (
                <Option key={type.value} value={type.value}>{type.label}</Option>
              ))}
            </Select>
          </Col>
        </Row>
        
        <Row style={{ marginTop: 20 }}>
          <Col span={24}>
            <div style={{ height: '500px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              {loading ? (
                <Spin size="large" />
              ) : keywords.length > 0 ? (
                <ReactECharts
                  option={getWordCloudOption()}
                  style={{ height: '100%', width: '100%' }}
                  notMerge={true}
                  lazyUpdate={true}
                />
              ) : (
                <Empty description="请选择分析条件" />
              )}
            </div>
          </Col>
        </Row>
      </Card>
    </div>
  );
}

export default WordCloud; 