# In Section 10.2, when asked about the capital of France, Claude wants to use
# a calculator to evaluate 1+1. This scratchbook is to debug this issue.
#
# Note that a better way to use tools seems to have been added to the API.
# In particular, client.messages.create() now has:
#   * 'tools' input parameter,
#   * 'tool_use' possible value of 'stop_reason' key of return value,
#   * 'server_tool_use' key of return value.
#
# The result of debugging is that
#   1. claude-3-haiku-20240307 wants us to calculate 1+1, but
#      claude-3-5-haiku-20241022 does not.
#   2. claude-3-haiku-20240307 can be convinced not to invoke a tool by
#      specifying that in the user prompt (in addition to the system prompt,
#      which already includes it.)

import os
import re
import anthropic

API_KEY = os.environ["ANTHROPIC_API_KEY"]

MODEL_NAME = "claude-3-haiku-20240307"
# MODEL_NAME = "claude-3-5-sonnet-20240620"
# MODEL_NAME = "claude-3-5-haiku-20241022"
# MODEL_NAME = "claude-3-5-sonnet-20241022"
# MODEL_NAME = "claude-3-7-sonnet-20250219"
# MODEL_NAME = "claude-sonnet-4-20250514"

system_prompt_tools_general_explanation = """You have access to a set of functions you can use to answer the user's question. This includes access to a
sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external
resources, except by invoking the below functions. You MUST NOT invoke a function if it is not required for your response.

You can invoke one or more functions by writing a "<function_calls>" block like the following as part of your
reply to the user:
<function_calls>
<invoke name="$FUNCTION_NAME">
<antml:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
</invoke>
<invoke name="$FUNCTION_NAME2">
...
</invoke>
</function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that
spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular
expressions.

The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of
your reply to the user.
You may then continue composing the rest of your reply to the user, respond to any errors, or make further function
calls as appropriate.
If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not
recognized as a call."""

system_prompt_tools_specific_tools = """Here are the functions available in JSONSchema format:
<tools>
<tool_description>
<tool_name>calculator</tool_name>
<description>
Calculator function for doing basic arithmetic.
Supports addition, subtraction, multiplication
</description>
<parameters>
<parameter>
<name>first_operand</name>
<type>int</type>
<description>First operand (before the operator)</description>
</parameter>
<parameter>
<name>second_operand</name>
<type>int</type>
<description>Second operand (after the operator)</description>
</parameter>
<parameter>
<name>operator</name>
<type>str</type>
<description>The operation to perform. Must be either +, -, *, or /</description>
</parameter>
</parameters>
</tool_description>
</tools>
"""

system_prompt = (
    system_prompt_tools_general_explanation + system_prompt_tools_specific_tools
)

stop_sequences = ["</function_calls>"]

client = anthropic.Anthropic(api_key=API_KEY)


def tool_use_example():
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": "Multiply 1,984,135 by 9,343,116"}],
        stop_sequences=stop_sequences,
    )

    assert message.stop_reason == "stop_sequence"
    assert message.stop_sequence == "</function_calls>"

    first_response_text = message.content[0].text

    pattern = (
        '<function_calls>.*<invoke name="calculator">.*'
        '<parameter name="first_operand">1984135</parameter>.*'
        '<parameter name="second_operand">9343116</parameter>.*'
        '<parameter name="operator">[*]</parameter>.*</invoke>'
    )

    assert re.search(pattern, first_response_text, re.DOTALL)

    function_result = """<function_results>
<result>
<tool_name>calculator</tool_name>
<stdout>
18538003464660
</stdout>
</result>
</function_results>"""

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.3,
        system=system_prompt,
        messages=[
            {"role": "user", "content": "Multiply 1,984,135 by 9,343,116"},
            {
                "role": "assistant",
                "content": first_response_text + "</function_calls>" + function_result,
            },
        ],
        stop_sequences=stop_sequences,
    )

    assert message.stop_reason == "end_turn"

    return message.content[0].text


def capital_of_france_example():
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.3,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    "Tell me the capital of France."  #
                    # Uncomment the following line to convince
                    # claude-3-haiku-20240307 not to use the calculator.
                    + "Do not invoke any tools unless necessary for the answer."  #
                ),
            }
        ],
        stop_sequences=stop_sequences,
    )

    return message


print(tool_use_example())
print(capital_of_france_example())
