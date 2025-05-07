import React, { useState, useEffect } from 'react';
import { Card, Select, Row, Col, Table, Tag, Typography, Space } from 'antd';
import { CarOutlined, LikeOutlined, CommentOutlined, InteractionOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';

const { Option } = Select;
const { Title, Text } = Typography;

function SingleCar() {
  const [cars, setCars] = useState([]);
  const [selectedCar, setSelectedCar] = useState(null);
  const [evaluationStats, setEvaluationStats] = useState(null);
  const [comments, setComments] = useState([]);
  const [interactionType, setInteractionType] = useState('0'); // 0: 点赞, 1: 评论, 2: 总互动
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    // 获取车型列表
    axios.get('http://localhost:5000/api/cars')
      .then(response => {
        setCars(response.data);
        if (response.data.length > 0) {
          setSelectedCar(response.data[0].car_id);
        }
      })
      .catch(error => console.error('Error fetching cars:', error))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedCar) {
      // 获取评价统计
      axios.get(`http://localhost:5000/api/car/${selectedCar}/evaluation-stats`)
        .then(response => {
          setEvaluationStats(response.data);
        })
        .catch(error => console.error('Error fetching evaluation stats:', error));

      // 获取评论数据
      axios.get(`http://localhost:5000/api/car/${selectedCar}/comments/${interactionType}`)
        .then(response => {
          setComments(response.data);
        })
        .catch(error => console.error('Error fetching comments:', error));
    }
  }, [selectedCar, interactionType]);

  // 饼图配置
  const getPieOption = () => {
    if (!evaluationStats) return {};
    
    const total = evaluationStats.positive_evaluation_count + 
                 evaluationStats.negative_evaluation_count + 
                 evaluationStats.neutral_evaluation_count + 
                 evaluationStats.irrelevant_evaluation_count;

    const data = [
      { 
        value: evaluationStats.positive_evaluation_count, 
        name: '正面评价',
        percentage: ((evaluationStats.positive_evaluation_count / total) * 100).toFixed(2),
        itemStyle: { color: '#52c41a' }  // 绿色
      },
      { 
        value: evaluationStats.negative_evaluation_count, 
        name: '负面评价',
        percentage: ((evaluationStats.negative_evaluation_count / total) * 100).toFixed(2),
        itemStyle: { color: '#f5222d' }  // 红色
      },
      { 
        value: evaluationStats.neutral_evaluation_count, 
        name: '中性评价',
        percentage: ((evaluationStats.neutral_evaluation_count / total) * 100).toFixed(2),
        itemStyle: { color: '#1890ff' }  // 蓝色
      },
      { 
        value: evaluationStats.irrelevant_evaluation_count, 
        name: '无关评价',
        percentage: ((evaluationStats.irrelevant_evaluation_count / total) * 100).toFixed(2),
        itemStyle: { color: '#d9d9d9' }  // 灰色
      }
    ];

    return {
      title: {
        text: '评价情感分布',
        left: 'center',
        top: 20,
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderColor: '#eee',
        borderWidth: 1,
        padding: [10, 15],
        textStyle: {
          color: '#666',
          fontSize: 14
        },
        formatter: function(params) {
          const colorSpan = `<span style="display:inline-block;margin-right:4px;border-radius:50%;width:10px;height:10px;background-color:${params.color}"></span>`;
          return `${colorSpan}<strong>${params.name}</strong><br/>` +
                 `数量：<strong>${params.value}</strong><br/>` +
                 `占比：<strong>${params.data.percentage}%</strong>`;
        }
      },
      legend: {
        orient: 'vertical',
        left: '5%',
        top: 'middle',
        itemGap: 20,
        textStyle: {
          fontSize: 14
        }
      },
      series: [
        {
          name: '评价类型',
          type: 'pie',
          radius: ['45%', '75%'],
          center: ['60%', '50%'],
          avoidLabelOverlap: true,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: true,
            formatter: '{b}\n{d}%',
            fontSize: 14,
            fontWeight: 'bold'
          },
          labelLine: {
            length: 15,
            length2: 10,
            smooth: true
          },
          emphasis: {
            label: {
              fontSize: 16,
              fontWeight: 'bold'
            },
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          },
          data: data
        }
      ]
    };
  };

  // 表格列配置
  const columns = [
    {
      title: '评论内容',
      dataIndex: 'comment',
      key: 'comment',
      width: '70%',
      render: (text) => (
        <div style={{ 
          padding: '12px',
          backgroundColor: '#f5f5f5',
          borderRadius: '6px',
          fontSize: '14px',
          lineHeight: '1.5'
        }}>
          {text}
        </div>
      )
    },
    {
      title: '互动量',
      dataIndex: 'interaction_count',
      key: 'interaction_count',
      width: '30%',
      align: 'center',
      render: (count) => (
        <Tag color="blue" style={{ 
          padding: '4px 12px',
          fontSize: '14px',
          borderRadius: '12px'
        }}>
          {count}
        </Tag>
      ),
      sorter: (a, b) => a.interaction_count - b.interaction_count,
    }
  ];

  const interactionOptions = [
    { value: '0', label: '点赞量', icon: <LikeOutlined /> },
    { value: '1', label: '评论量', icon: <CommentOutlined /> },
    { value: '2', label: '总互动量', icon: <InteractionOutlined /> }
  ];

  return (
    <div>
      <Card 
        title={<Title level={4}>单车型数据分析</Title>}
        className="analysis-card"
      >
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card size="small" className="filter-card">
              <Space size={20}>
                <Select
                  showSearch
                  style={{ width: 300 }}
                  placeholder="搜索并选择车型"
                  value={selectedCar}
                  onChange={setSelectedCar}
                  loading={loading}
                  optionFilterProp="children"
                  filterOption={(input, option) =>
                    (option?.children ?? '').toLowerCase().includes(input.toLowerCase())
                  }
                  suffixIcon={<CarOutlined style={{ fontSize: '18px' }} />}
                  className="custom-select large-select"
                  size="large"
                >
                  {cars.map(car => (
                    <Option key={car.car_id} value={car.car_id}>
                      <Space>
                        <CarOutlined style={{ fontSize: '16px' }} />
                        <span style={{ fontSize: '15px' }}>{car.car_name}</span>
                      </Space>
                    </Option>
                  ))}
                </Select>

                <Select
                  style={{ width: 250 }}
                  placeholder="选择互动类型"
                  value={interactionType}
                  onChange={setInteractionType}
                  className="custom-select large-select"
                  size="large"
                >
                  {interactionOptions.map(option => (
                    <Option key={option.value} value={option.value}>
                      <Space>
                        {React.cloneElement(option.icon, { style: { fontSize: '16px' } })}
                        <span style={{ fontSize: '15px' }}>{option.label}</span>
                      </Space>
                    </Option>
                  ))}
                </Select>
              </Space>
            </Card>
          </Col>
        </Row>
        
        <Row gutter={[16, 16]} style={{ marginTop: 20 }}>
          <Col span={12}>
            <Card 
              className="chart-card"
              bodyStyle={{ height: '450px', padding: '20px' }}
            >
              <ReactECharts 
                option={getPieOption()} 
                style={{ height: '100%' }}
                notMerge={true}
                lazyUpdate={true}
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card 
              className="table-card"
              bodyStyle={{ padding: '0' }}
            >
              <Table
                dataSource={comments}
                columns={columns}
                pagination={{
                  pageSize: 10,
                  showTotal: (total) => `共 ${total} 条评论`,
                  showSizeChanger: false,
                  style: { padding: '16px' }
                }}
                rowKey={(record, index) => index}
                style={{ 
                  backgroundColor: '#fff',
                  borderRadius: '8px',
                }}
              />
            </Card>
          </Col>
        </Row>
      </Card>
      <style jsx global>{`
        .analysis-card {
          background: #fff;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .filter-card {
          background: #fafafa;
          border-radius: 6px;
          padding: 24px;
        }
        .custom-select .ant-select-selector {
          border-radius: 8px !important;
          border: 1px solid #d9d9d9 !important;
          transition: all 0.3s !important;
        }
        .large-select.ant-select-lg .ant-select-selector {
          height: 45px !important;
          padding: 4px 16px !important;
        }
        .large-select.ant-select-lg .ant-select-selection-item {
          line-height: 35px !important;
          font-size: 15px !important;
        }
        .large-select.ant-select-lg .ant-select-selection-placeholder {
          line-height: 35px !important;
          font-size: 15px !important;
        }
        .custom-select:hover .ant-select-selector {
          border-color: #40a9ff !important;
        }
        .custom-select.ant-select-focused .ant-select-selector {
          border-color: #40a9ff !important;
          box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2) !important;
        }
        .custom-select .ant-select-selection-placeholder {
          color: #bfbfbf;
        }
        .custom-select .ant-select-arrow {
          color: #8c8c8c;
          margin-top: -10px !important;
        }
        .chart-card, .table-card {
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .ant-table-thead > tr > th {
          background: #fafafa;
          padding: 16px 12px;
        }
        .ant-table-tbody > tr > td {
          padding: 12px;
        }
        .ant-table-tbody > tr:hover > td {
          background: #f0f5ff !important;
        }
      `}</style>
    </div>
  );
}

export default SingleCar; 