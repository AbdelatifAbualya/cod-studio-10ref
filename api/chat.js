module.exports = async (req, res) => {
  // Model configuration
  const FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v3-0324";
  
  // Handle CORS for all requests
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST requests for the main functionality
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Get Fireworks API key from environment variables
    const apiKey = process.env.FIREWORKS_API_KEY;
    if (!apiKey) {
      console.error('FIREWORKS_API_KEY environment variable not set');
      return res.status(500).json({ 
        error: 'Server configuration error',
        message: 'API key not configured. Please check server environment variables.' 
      });
    }

    // Extract the request body
    const { 
      model, 
      messages, 
      temperature, 
      top_p, 
      top_k, 
      max_tokens, 
      presence_penalty, 
      frequency_penalty, 
      stream, 
      tools, 
      tool_choice,
      enhanced_cod_mode = false,
      reasoning_method = "cod"
    } = req.body;

    // Validate required fields
    if (!model || !messages) {
      console.error('Missing required fields in request:', { model: !!model, messages: !!messages });
      return res.status(400).json({ 
        error: 'Bad request',
        message: 'Missing required fields: model and messages' 
      });
    }

    console.log('Processing request:', { 
      model, 
      messageCount: messages.length, 
      stream: !!stream,
      enhanced_cod_mode,
      reasoning_method,
      toolsEnabled: !!(tools && tools.length > 0)
    });

    // Enhanced CoD Mode: Two-Stage API Processing
    if (enhanced_cod_mode && reasoning_method === "cod") {
      return await handleEnhancedCoD(req, res, {
        apiKey,
        model: FIREWORKS_MODEL, // Use the constant model name
        messages,
        temperature,
        top_p,
        top_k,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        stream,
        tools,
        tool_choice
      });
    }

    // Standard single API call (existing functionality)
    return await handleStandardRequest(req, res, {
      apiKey,
      model: FIREWORKS_MODEL, // Use the constant model name
      messages,
      temperature,
      top_p,
      top_k,
      max_tokens,
      presence_penalty,
      frequency_penalty,
      stream,
      tools,
      tool_choice
    });

  } catch (error) {
    console.error('Server error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
};

// Enhanced CoD: Two-stage API processing
async function handleEnhancedCoD(req, res, params) {
  const {
    apiKey,
    model,
    messages,
    temperature,
    top_p,
    top_k,
    max_tokens,
    presence_penalty,
    frequency_penalty,
    stream,
    tools,
    tool_choice
  } = params;

  try {
    console.log('Starting Enhanced CoD two-stage processing...');

    // STAGE 1: Initial CoD Reasoning with Deep Reflections
    console.log('Stage 1: Initial CoD reasoning...');
    const stage1Response = await performStage1Reasoning({
      apiKey,
      model,
      messages,
      temperature,
      top_p,
      top_k,
      max_tokens,
      presence_penalty,
      frequency_penalty,
      tools,
      tool_choice
    });

    if (!stage1Response.success) {
      throw new Error(`Stage 1 failed: ${stage1Response.error}`);
    }

    console.log('Stage 1 completed successfully');

    // STAGE 2: Final Verification and Answer Refinement
    console.log('Stage 2: Final verification and refinement...');
    const stage2Response = await performStage2Verification({
      apiKey,
      model,
      originalMessages: messages,
      stage1Content: stage1Response.content,
      stage1Thinking: stage1Response.thinking,
      stage1Answer: stage1Response.answer,
      temperature,
      top_p,
      top_k,
      max_tokens,
      presence_penalty,
      frequency_penalty,
      tools,
      tool_choice
    });

    if (!stage2Response.success) {
      throw new Error(`Stage 2 failed: ${stage2Response.error}`);
    }

    console.log('Enhanced CoD processing completed successfully');

    // Combine both stages for final response
    const enhancedResponse = {
      choices: [{
        message: {
          role: "assistant",
          content: stage2Response.finalContent,
          enhanced_cod_metadata: {
            stage1_thinking: stage1Response.thinking,
            stage1_answer: stage1Response.answer,
            stage2_verification: stage2Response.verification,
            stage2_final_answer: stage2Response.finalAnswer,
            total_stages: 2,
            reasoning_method: "enhanced_cod"
          }
        },
        finish_reason: "stop"
      }],
      usage: {
        prompt_tokens: (stage1Response.usage?.prompt_tokens || 0) + (stage2Response.usage?.prompt_tokens || 0),
        completion_tokens: (stage1Response.usage?.completion_tokens || 0) + (stage2Response.usage?.completion_tokens || 0),
        total_tokens: (stage1Response.usage?.total_tokens || 0) + (stage2Response.usage?.total_tokens || 0)
      },
      enhanced_cod: true
    };

    return res.status(200).json(enhancedResponse);

  } catch (error) {
    console.error('Enhanced CoD processing error:', error);
    return res.status(500).json({ 
      error: 'Enhanced CoD processing failed',
      message: error.message 
    });
  }
}

// Stage 1: Initial CoD Reasoning
async function performStage1Reasoning(params) {
  const {
    apiKey,
    model,
    messages,
    temperature,
    top_p,
    top_k,
    max_tokens,
    presence_penalty,
    frequency_penalty,
    tools,
    tool_choice
  } = params;

  const stage1SystemPrompt = `You are an AI assistant using Enhanced Chain of Draft (CoD) reasoning methodology. This is STAGE 1 of a two-stage enhanced reasoning process.

Your goal is to solve the given problem through structured Chain of Draft reasoning with deep reflections:

**STAGE 1 STRUCTURE:**
1. **Concise CoD Steps:** Use very short, information-dense steps (target 8-15 words per step). Focus on key calculations and essential intermediate results.

2. **Deep Analytical Reflections:** After every 2-3 CoD steps, insert a "DEEP REFLECTION X:" block (where X is 1, 2, or 3). You must include exactly three deep reflection blocks.
   - These reflections should be comprehensive and verbose (no word limit)
   - Verify intermediate results, assess confidence levels, identify potential pitfalls
   - Consider alternative approaches and validate assumptions
   - Check for logical consistency and mathematical accuracy

3. **Critical Analysis Points:** Throughout your reasoning, pay special attention to:
   - Accuracy of calculations and logical deductions
   - Completeness of the solution approach
   - Potential edge cases or special considerations
   - Confidence levels in different parts of your reasoning

**Output Format:**
- Start with concise CoD steps
- Intersperse with 3 DEEP REFLECTION blocks as specified
- End with "####" separator
- Provide a comprehensive preliminary answer after the separator

Remember: This is Stage 1. Focus on thorough analysis and reasoning. A Stage 2 verification will follow to finalize and refine your work.`;

  const stage1Messages = [
    { role: "system", content: stage1SystemPrompt },
    ...messages.filter(msg => msg.role !== "system")
  ];

  const stage1Payload = {
    model,
    messages: stage1Messages,
    temperature: temperature || 0.6,
    top_p: top_p || 1,
    top_k: top_k || 40,
    max_tokens: Math.min(max_tokens || 8192, 12000), // Allow more tokens for deep reflections
    presence_penalty: presence_penalty || 0,
    frequency_penalty: frequency_penalty || 0,
    stream: false // Always non-streaming for stage processing
  };

  if (tools && tools.length > 0) {
    stage1Payload.tools = tools;
    if (tool_choice) {
      stage1Payload.tool_choice = tool_choice;
    }
  }

  const response = await fetch('https://api.fireworks.ai/inference/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(stage1Payload)
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Stage 1 API Error:', response.status, errorText);
    return { success: false, error: `Stage 1 API error: ${response.status}` };
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content || "";
  
  // Parse CoD structure
  const separatorIndex = content.indexOf("####");
  let thinking = "", answer = "";
  
  if (separatorIndex !== -1) {
    thinking = content.substring(0, separatorIndex).trim();
    answer = content.substring(separatorIndex + 4).trim();
  } else {
    thinking = content;
    answer = "No preliminary answer found.";
  }

  return {
    success: true,
    content,
    thinking,
    answer,
    usage: data.usage
  };
}

// Stage 2: Final Verification and Refinement
async function performStage2Verification(params) {
  const {
    apiKey,
    model,
    originalMessages,
    stage1Content,
    stage1Thinking,
    stage1Answer,
    temperature,
    top_p,
    top_k,
    max_tokens,
    presence_penalty,
    frequency_penalty,
    tools,
    tool_choice
  } = params;

  const stage2SystemPrompt = `You are an AI assistant performing STAGE 2 of Enhanced Chain of Draft reasoning: Final Verification and Answer Refinement.

You will receive:
1. The original user question/problem
2. Stage 1 reasoning (CoD steps with deep reflections)
3. Stage 1 preliminary answer

**STAGE 2 OBJECTIVES:**
1. **Critical Verification:** Thoroughly review the Stage 1 reasoning for:
   - Mathematical accuracy and logical consistency
   - Completeness of the solution approach
   - Potential errors or oversights
   - Validity of assumptions made

2. **Quality Assessment:** Evaluate:
   - Whether all aspects of the original question are addressed
   - Confidence level in the reasoning chain
   - Alternative solution paths that might be more robust
   - Edge cases or special considerations

3. **Final Refinement:** Based on your analysis:
   - Confirm the correctness of the Stage 1 answer, or
   - Identify and correct any errors found, or
   - Enhance the answer with additional insights

**Output Structure:**
```
STAGE 2 VERIFICATION:
[Your comprehensive verification analysis of Stage 1 reasoning]

FINAL REFLECTION:
[Your assessment of the solution quality, confidence level, and any refinements needed]

####

ENHANCED FINAL ANSWER:
[Your final, refined answer that incorporates all insights from both stages]
```

Focus on producing the highest quality, most accurate final answer possible.`;

  const stage2Messages = [
    { role: "system", content: stage2SystemPrompt },
    { role: "user", content: `ORIGINAL PROBLEM:
${originalMessages[originalMessages.length - 1]?.content || "No user message found"}

STAGE 1 REASONING:
${stage1Thinking}

STAGE 1 PRELIMINARY ANSWER:
${stage1Answer}

Please perform Stage 2 verification and provide the enhanced final answer.` }
  ];

  const stage2Payload = {
    model,
    messages: stage2Messages,
    temperature: Math.max(0.1, (temperature || 0.6) * 0.7), // Lower temperature for verification
    top_p: top_p || 1,
    top_k: top_k || 40,
    max_tokens: Math.min(max_tokens || 8192, 8192),
    presence_penalty: presence_penalty || 0,
    frequency_penalty: frequency_penalty || 0,
    stream: false
  };

  if (tools && tools.length > 0) {
    stage2Payload.tools = tools;
    if (tool_choice) {
      stage2Payload.tool_choice = tool_choice;
    }
  }

  const response = await fetch('https://api.fireworks.ai/inference/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(stage2Payload)
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Stage 2 API Error:', response.status, errorText);
    return { success: false, error: `Stage 2 API error: ${response.status}` };
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content || "";
  
  // Parse Stage 2 structure
  const finalSeparatorIndex = content.indexOf("####");
  let verification = "", finalAnswer = "";
  
  if (finalSeparatorIndex !== -1) {
    verification = content.substring(0, finalSeparatorIndex).trim();
    finalAnswer = content.substring(finalSeparatorIndex + 4).trim();
  } else {
    verification = content;
    finalAnswer = "Verification completed but no final answer section found.";
  }

  // Combine everything for final output
  const finalContent = `${stage1Thinking}

======= ENHANCED VERIFICATION & REFINEMENT =======

${verification}

####

${finalAnswer}`;

  return {
    success: true,
    verification,
    finalAnswer,
    finalContent,
    usage: data.usage
  };
}

// Standard single API call (existing functionality)
async function handleStandardRequest(req, res, params) {
  const {
    apiKey,
    model,
    messages,
    temperature,
    top_p,
    top_k,
    max_tokens,
    presence_penalty,
    frequency_penalty,
    stream,
    tools,
    tool_choice
  } = params;

  const fireworksPayload = {
    model,
    messages,
    temperature: temperature || 0.6,
    top_p: top_p || 1,
    top_k: top_k || 40,
    max_tokens: max_tokens || 25000,
    presence_penalty: presence_penalty || 0,
    frequency_penalty: frequency_penalty || 0,
    stream: stream || false
  };

  // Add tools if provided
  if (tools && tools.length > 0) {
    fireworksPayload.tools = tools;
    if (tool_choice) {
      fireworksPayload.tool_choice = tool_choice;
    }
  }

  const fireworksHeaders = {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
  };

  // Handle streaming responses
  if (stream) {
    fireworksHeaders['Accept'] = 'text/event-stream';
    
    const response = await fetch('https://api.fireworks.ai/inference/v1/chat/completions', {
      method: 'POST',
      headers: fireworksHeaders,
      body: JSON.stringify(fireworksPayload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Fireworks API Error:', response.status, errorText);
      return res.status(response.status).json({ 
        error: 'API request failed',
        message: errorText 
      });
    }

    // Set headers for streaming
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (!response.body) {
      return res.status(500).json({ error: 'No response body from API' });
    }

    // Handle streaming with proper async iteration
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        res.write(chunk);
      }
      res.end();
    } catch (error) {
      console.error('Streaming error:', error);
      res.write(`data: {"error": "Streaming interrupted"}\n\n`);
      res.end();
    }

  } else {
    // Handle non-streaming responses
    const response = await fetch('https://api.fireworks.ai/inference/v1/chat/completions', {
      method: 'POST',
      headers: fireworksHeaders,
      body: JSON.stringify(fireworksPayload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Fireworks API Error:', response.status, errorText);
      return res.status(response.status).json({ 
        error: 'API request failed',
        message: errorText 
      });
    }

    const data = await response.json();
    return res.status(200).json(data);
  }
}
