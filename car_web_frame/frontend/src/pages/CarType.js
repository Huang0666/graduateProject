import React, { useState, useEffect } from 'react';
import { Card, Select, Row, Col, Table, Empty } from 'antd';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';

const { Option } = Select;

const carTypes = ['燃油', '新能源', 'SUV', 'MPV', '轿车'];

function CarType() {
  const [selectedType, setSelectedType] = useState('燃油');
  const [interactionType, setInteractionType] = useState('0');
  const [comments, setComments] = useState([]);
  const [evaluationStats, setEvaluationStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 10,
    total: 0
  });

  const fetchComments = async (page = 1, pageSize = 10) => {
    if (!selectedType) return;

    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/api/car-type/comments', {
        params: {
          category: selectedType,
          interaction_type: interactionType,
          page: page,
          page_size: pageSize
        }
      });

      setComments(response.data.data);
      setPagination({
        current: response.data.page,
        pageSize: response.data.page_size,
        total: response.data.total
      });
    } catch (error) {
      console.error('Error fetching comments:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchEvaluationStats = async () => {
    if (!selectedType) return;

    try {
      const response = await axios.get('http://localhost:5000/api/car-type/evaluation-stats', {
        params: {
          category: selectedType
        }
      });
      setEvaluationStats(response.data);
    } catch (error) {
      console.error('Error fetching evaluation stats:', error);
    }
  };

  useEffect(() => {
    if (selectedType) {
      fetchComments(1);
      fetchEvaluationStats();
    }
  }, []);

  useEffect(() => {
    if (selectedType) {
      fetchComments(1);
    }
  }, [interactionType]);

  const handleTableChange = (pagination) => {
    fetchComments(pagination.current, pagination.pageSize);
  };

  const getPieOption = () => {
    if (!evaluationStats) return {};
    
    const total = evaluationStats.total_positive + 
                 evaluationStats.total_negative + 
                 evaluationStats.total_neutral + 
                 evaluationStats.total_irrelevant;

    const data = [
      { 
        value: evaluationStats.total_positive, 
        name: '正面评价',
        percentage: ((evaluationStats.total_positive / total) * 100).toFixed(2)
      },
      { 
        value: evaluationStats.total_negative, 
        name: '负面评价',
        percentage: ((evaluationStats.total_negative / total) * 100).toFixed(2)
      },
      { 
        value: evaluationStats.total_neutral, 
        name: '中性评价',
        percentage: ((evaluationStats.total_neutral / total) * 100).toFixed(2)
      },
      { 
        value: evaluationStats.total_irrelevant, 
        name: '无关评价',
        percentage: ((evaluationStats.total_irrelevant / total) * 100).toFixed(2)
      }
    ];

    return {
      title: {
        text: `${selectedType}类型车辆评价分布`,
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: function(params) {
          return `${params.name}<br/>数量: ${params.value}<br/>占比: ${params.data.percentage}%`;
        }
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        top: 'middle',
        data: ['正面评价', '负面评价', '中性评价', '无关评价']
      },
      series: [
        {
          name: '评价类型',
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: true,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: true,
            formatter: '{b}: {c} ({d}%)'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '16',
              fontWeight: 'bold'
            }
          },
          data: data
        }
      ]
    };
  };

  const columns = [
    {
      title: '评论内容',
      dataIndex: 'content',
      key: 'content',
      width: '70%',
    },
    {
      title: '互动量',
      dataIndex: 'interaction_count',
      key: 'interaction_count',
      width: '30%',
      sorter: (a, b) => a.interaction_count - b.interaction_count,
    }
  ];

  const getInteractionTypeLabel = () => {
    switch (interactionType) {
      case '0': return '点赞量';
      case '1': return '评论量';
      case '2': return '总互动量';
      default: return '';
    }
  };

  return (
    <div>
      <Card title="车型类别分析">
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Select
              style={{ width: 200, marginRight: 16 }}
              placeholder="选择车型类别"
              value={selectedType}
              onChange={setSelectedType}
            >
              {carTypes.map(type => (
                <Option key={type} value={type}>{type}</Option>
              ))}
            </Select>
            
            <Select
              style={{ width: 200 }}
              placeholder="选择互动类型"
              value={interactionType}
              onChange={setInteractionType}
            >
              <Option value="0">点赞量</Option>
              <Option value="1">评论量</Option>
              <Option value="2">总互动量</Option>
            </Select>
          </Col>
        </Row>
        
        <Row gutter={[16, 16]} style={{ marginTop: 20 }}>
          <Col span={12}>
            {selectedType && evaluationStats ? (
              <Card>
                <ReactECharts 
                  option={getPieOption()} 
                  style={{ height: '400px' }}
                  notMerge={true}
                  lazyUpdate={true}
                />
              </Card>
            ) : (
              <Empty description={selectedType ? "加载中..." : "请选择车型类别"} />
            )}
          </Col>
          <Col span={12}>
            {selectedType ? (
              <Table
                columns={columns}
                dataSource={comments}
                pagination={pagination}
                onChange={handleTableChange}
                loading={loading}
                rowKey={(record, index) => index}
                title={() => (
                  <div style={{ fontWeight: 'bold' }}>
                    {`${selectedType}类型车辆 - 按${getInteractionTypeLabel()}排序的评论`}
                  </div>
                )}
              />
            ) : (
              <Empty description="请选择车型类别" />
            )}
          </Col>
        </Row>
      </Card>
    </div>
  );
}

export default CarType; 