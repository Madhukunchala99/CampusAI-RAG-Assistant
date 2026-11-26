
import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import {
  Sparkles, Plus, Search, MessageSquare, BookOpen,
  Settings, GraduationCap, ArrowUp, Paperclip, Menu,
  X, Copy, Check, ShieldCheck, Zap, FileText,
  Database, CheckCircle2, Clock, Trash2, Image,
  Upload, FileUp, ChevronDown
} from 'lucide-react'
import './App.css'


// TYPES

type SourceDetail = {
  filename: string
  page: number
  chunk_id: number
  text: string
}

type Message = {
  id: number
  role: 'user' | 'assistant'
  content: string
  source_details?: SourceDetail[]
}

type Conversation = {
  id: string
  title: string
  messages: Message[]
  updatedAt: number
}

type SavedState = {
  conversations: Conversation[]
  activeChatId: string | null
}

type DocumentInfo = {
  filename: string
  indexed_pages: number
}

const API_URL = 'http://127.0.0.1:8000'
const STORAGE_KEY = 'campus-ai-conversations'


// SUGGESTIONS

const suggestions = [
  {
    icon: BookOpen,
    title: 'College regulations',
    description: 'Understand attendance, exams and credits.',
    prompt: 'What are the college attendance regulations?'
  },
  {
    icon: GraduationCap,
    title: 'Academic information',
    description: 'Explore academic policies and requirements.',
    prompt: 'Explain the examination policy.'
  },
  {
    icon: ShieldCheck,
    title: 'Student guidelines',
    description: 'Find important student rules and guidelines.',
    prompt: 'What happens if attendance is below 75%?'
  }
]


// LOAD SAVED CONVERSATIONS

function loadSavedState(): SavedState {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)

    if (saved) {
      const parsed = JSON.parse(saved) as SavedState

      if (Array.isArray(parsed.conversations)) {
        return {
          conversations: parsed.conversations,
          activeChatId: parsed.activeChatId ?? null
        }
      }
    }
  } catch (error) {
    console.error('Could not load saved conversations:', error)
  }

  return {
    conversations: [],
    activeChatId: null
  }
}


// MAIN APP

function App() {
  const [savedState] = useState(loadSavedState)

  const [conversations, setConversations] = useState<Conversation[]>(
    savedState.conversations
  )

  const [activeChatId, setActiveChatId] = useState<string | null>(
    savedState.activeChatId
  )

  const [question, setQuestion] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [copiedId, setCopiedId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activePage, setActivePage] = useState<'chat' | 'knowledge'>('chat')
  const [searchTerm, setSearchTerm] = useState('')

  // Attachment states

  const [attachmentMenuOpen, setAttachmentMenuOpen] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)

  const [uploadingPdf, setUploadingPdf] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')
  const [uploadError, setUploadError] = useState('')

  // Knowledge base states

  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loadingDocuments, setLoadingDocuments] = useState(false)
  const [documentError, setDocumentError] = useState('')

  // File input references

  const pdfInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  // Automatic scrolling reference

  const messagesEndRef = useRef<HTMLDivElement>(null)

  const activeConversation = conversations.find(
    conversation => conversation.id === activeChatId
  )

  const messages = activeConversation?.messages ?? []


  // SAVE CONVERSATIONS

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          conversations,
          activeChatId
        })
      )
    } catch (error) {
      console.error('Could not save conversations:', error)
    }
  }, [conversations, activeChatId])


  // AUTOMATICALLY SCROLL TO LATEST MESSAGE

  useEffect(() => {
    if (activePage === 'chat') {
      messagesEndRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'end'
      })
    }
  }, [messages, isLoading, activePage])


  // CLEAN UP IMAGE PREVIEW URL

  useEffect(() => {
    return () => {
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview)
      }
    }
  }, [imagePreview])


  // LOAD DOCUMENTS FROM BACKEND

  const fetchDocuments = async () => {
    setLoadingDocuments(true)
    setDocumentError('')

    try {
      const response = await fetch(`${API_URL}/documents`)

      if (!response.ok) {
        throw new Error('Could not load documents.')
      }

      const data = await response.json()

      setDocuments(data.documents ?? [])

    } catch (error) {
      console.error('Document loading error:', error)

      setDocumentError(
        error instanceof Error
          ? error.message
          : 'Could not connect to backend.'
      )

    } finally {
      setLoadingDocuments(false)
    }
  }


  // LOAD DOCUMENTS WHEN KNOWLEDGE PAGE OPENS

  useEffect(() => {
    if (activePage === 'knowledge') {
      fetchDocuments()
    }
  }, [activePage])


  // NEW CHAT

  const newChat = () => {
    if (isLoading || uploadingPdf) return

    setActiveChatId(null)
    setActivePage('chat')
    setQuestion('')
    setSidebarOpen(false)
    removeImage()
  }


  // OPEN CONVERSATION

  const openConversation = (id: string) => {
    if (isLoading) return

    setActiveChatId(id)
    setActivePage('chat')
    setQuestion('')
    setSidebarOpen(false)
    removeImage()
  }


  // REMOVE SELECTED IMAGE

  const removeImage = () => {
    setSelectedImage(null)
    setImagePreview(null)

    if (imageInputRef.current) {
      imageInputRef.current.value = ''
    }
  }


  // SELECT IMAGE

  const handleImageSelection = (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]

    if (!file) return

    const allowedTypes = [
      'image/jpeg',
      'image/png',
      'image/webp'
    ]

    if (!allowedTypes.includes(file.type)) {
      setUploadError('Please select a JPG, PNG or WebP image.')
      event.target.value = ''
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setUploadError('Image size cannot exceed 10 MB.')
      event.target.value = ''
      return
    }

    setUploadError('')
    setUploadMessage('')

    setSelectedImage(file)
    setImagePreview(URL.createObjectURL(file))
    setAttachmentMenuOpen(false)
  }


  // UPLOAD PDF

  const handlePdfSelection = async (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]

    if (!file) return

    event.target.value = ''

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Please select a PDF file.')
      return
    }

    if (file.size > 20 * 1024 * 1024) {
      setUploadError('PDF size cannot exceed 20 MB.')
      return
    }

    setUploadingPdf(true)
    setUploadError('')
    setUploadMessage('Uploading and indexing PDF...')
    setAttachmentMenuOpen(false)

    try {
      const formData = new FormData()

      formData.append('file', file)

      const response = await fetch(`${API_URL}/upload-pdf`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'PDF upload failed.')
      }

      setUploadMessage(
        `${data.filename} uploaded successfully and indexed.`
      )

      await fetchDocuments()

    } catch (error) {
      console.error('PDF upload error:', error)

      setUploadMessage('')

      setUploadError(
        error instanceof Error
          ? error.message
          : 'Could not upload PDF.'
      )

    } finally {
      setUploadingPdf(false)
    }
  }


  // ADD MESSAGE TO CONVERSATION

  const addMessageToConversation = (
    chatId: string,
    message: Message
  ) => {
    setConversations(previous => {

      const existing = previous.find(
        conversation => conversation.id === chatId
      )

      if (existing) {
        return previous.map(conversation =>
          conversation.id === chatId
            ? {
                ...conversation,
                messages: [...conversation.messages, message],
                updatedAt: Date.now()
              }
            : conversation
        )
      }

      const newConversation: Conversation = {
        id: chatId,
        title: message.role === 'user'
          ? message.content.slice(0, 42)
          : 'New conversation',
        messages: [message],
        updatedAt: Date.now()
      }

      return [newConversation, ...previous]
    })
  }


  // SEND MESSAGE

  const sendMessage = async (value = question) => {
    const trimmed = value.trim()

    if ((!trimmed && !selectedImage) || isLoading || uploadingPdf) {
      return
    }

    const currentChatId = activeChatId ?? String(Date.now())

    const imageToSend = selectedImage

    const userContent = imageToSend
      ? `${trimmed || 'Please analyze this image.'}\n\n[Image attached: ${imageToSend.name}]`
      : trimmed

    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: userContent
    }

    addMessageToConversation(currentChatId, userMessage)

    setActiveChatId(currentChatId)
    setQuestion('')
    setIsLoading(true)

    removeImage()

    try {
      let response: Response

      if (imageToSend) {
        const formData = new FormData()

        formData.append(
          'question',
          trimmed || 'Describe this image.'
        )

        formData.append('image', imageToSend)

        response = await fetch(`${API_URL}/chat-image`, {
          method: 'POST',
          body: formData
        })

      } else {
        response = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            question: trimmed
          })
        })
      }

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get an answer.')
      }

      const assistantMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.answer,
        source_details: data.source_details ?? []
      }

      addMessageToConversation(currentChatId, assistantMessage)

    } catch (error) {
      console.error('Chat API error:', error)

      const errorMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: error instanceof Error
          ? `Sorry, something went wrong: ${error.message}`
          : 'Could not connect to the server.'
      }

      addMessageToConversation(currentChatId, errorMessage)

    } finally {
      setIsLoading(false)
    }
  }


  // DELETE CONVERSATION

  const deleteConversation = (id: string) => {
    if (isLoading) return

    setConversations(previous =>
      previous.filter(conversation => conversation.id !== id)
    )

    if (activeChatId === id) {
      setActiveChatId(null)
    }
  }


  // COPY MESSAGE

  const copyMessage = async (message: Message) => {
    try {
      await navigator.clipboard.writeText(message.content)

      setCopiedId(message.id)

      window.setTimeout(() => setCopiedId(null), 1800)

    } catch {
      setCopiedId(null)
    }
  }


  // SEARCH CONVERSATIONS

  const filteredConversations = conversations.filter(conversation =>
    conversation.title.toLowerCase().includes(searchTerm.toLowerCase())
  )


  // RENDER

  return (
    <div className="app-shell">

      {/* HIDDEN FILE INPUTS */}

      <input
        ref={pdfInputRef}
        type="file"
        accept=".pdf,application/pdf"
        hidden
        onChange={handlePdfSelection}
      />

      <input
        ref={imageInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        hidden
        onChange={handleImageSelection}
      />


      {/* MOBILE OVERLAY */}

      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            className="mobile-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>


      {/* SIDEBAR */}

      <motion.aside
        className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}
        initial={false}
      >

        <div className="brand">
          <div className="brand-icon">
            <GraduationCap size={23} />
          </div>

          <div className="brand-text">
            <h2>Campus<span>AI</span></h2>
            <p>YOUR COLLEGE COMPANION</p>
          </div>

          <button
            className="icon-button mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={19} />
          </button>
        </div>


        <button className="new-chat-button" onClick={newChat}>
          <Plus size={18} />
          <span>New conversation</span>
          <span className="shortcut">⌘ K</span>
        </button>


        <div className="sidebar-search">
          <Search size={16} />

          <input
            placeholder="Search conversations"
            value={searchTerm}
            onChange={event => setSearchTerm(event.target.value)}
          />
        </div>


        {/* WORKSPACE NAVIGATION */}

        <div className="sidebar-section">
          <p className="sidebar-label">WORKSPACE</p>

          <button
            className={`nav-item ${activePage === 'chat' ? 'active' : ''}`}
            onClick={() => {
              setActivePage('chat')
              setSidebarOpen(false)
            }}
          >
            <MessageSquare size={17} />
            <span>AI Assistant</span>
          </button>

          <button
            className={`nav-item ${activePage === 'knowledge' ? 'active' : ''}`}
            onClick={() => {
              setActivePage('knowledge')
              setSidebarOpen(false)
            }}
          >
            <BookOpen size={17} />
            <span>Knowledge base</span>
          </button>
        </div>


        {/* CONVERSATION HISTORY */}

        <div className="sidebar-section history-section">

          <div className="history-heading">
            <p className="sidebar-label">RECENT CONVERSATIONS</p>
            <span className="history-count">{conversations.length}</span>
          </div>

          {filteredConversations.length > 0 ? (
            <div className="history-list">

              {filteredConversations.map(conversation => (
                <div
                  key={conversation.id}
                  className={`history-entry ${
                    activeChatId === conversation.id ? 'selected' : ''
                  }`}
                >

                  <button
                    className="history-item"
                    onClick={() => openConversation(conversation.id)}
                    disabled={isLoading}
                    title={conversation.title}
                  >
                    <MessageSquare size={15} />
                    <span>{conversation.title}</span>
                  </button>

                  <button
                    className="delete-chat-button"
                    onClick={() => deleteConversation(conversation.id)}
                    disabled={isLoading}
                    title="Delete conversation"
                    aria-label="Delete conversation"
                  >
                    <Trash2 size={14} />
                  </button>

                </div>
              ))}

            </div>
          ) : (
            <p className="empty-history">
              {searchTerm
                ? 'No matching conversations found.'
                : 'Your conversations will appear here.'}
            </p>
          )}

        </div>


        {/* SIDEBAR BOTTOM */}

        <div className="sidebar-bottom">

          <div className="plan-card">
            <div className="plan-icon">
              <Zap size={17} />
            </div>

            <div>
              <strong>AI-powered learning</strong>
              <p>Answers grounded in your documents</p>
            </div>
          </div>

          <button className="profile-button">
            <div className="avatar">H</div>

            <div className="profile-info">
              <strong>Student</strong>
              <span>Personal workspace</span>
            </div>

            <Settings size={17} />
          </button>

        </div>

      </motion.aside>


      {/* MAIN AREA */}

      <main className="main-area">

        {/* TOPBAR */}

        <header className="topbar">

          <div className="topbar-left">

            <button
              className="icon-button menu-button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
            >
              <Menu size={21} />
            </button>

            <div className="breadcrumb">
              <span>Workspace</span>
              <span className="breadcrumb-separator">/</span>

              <strong>
                {activePage === 'chat' ? 'AI Assistant' : 'Knowledge Base'}
              </strong>
            </div>

          </div>

          <div className="topbar-right">

            <div className="status-pill">
              <span className="status-dot" />

              <span>
                {isLoading
                  ? 'Thinking...'
                  : uploadingPdf
                    ? 'Uploading PDF...'
                    : 'Assistant ready'}
              </span>
            </div>

            <button className="icon-button" aria-label="Settings">
              <Settings size={19} />
            </button>

          </div>

        </header>


        {/* AI ASSISTANT PAGE */}

        {activePage === 'chat' && (

          <div className="chat-container">

            {messages.length === 0 ? (

              <motion.section
                className="welcome-section"
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.55 }}
              >

                <div className="welcome-orb">
                  <div className="orb-ring ring-one" />
                  <div className="orb-ring ring-two" />

                  <div className="orb-core">
                    <Sparkles size={30} />
                  </div>

                  <span className="orb-spark spark-one">✦</span>
                  <span className="orb-spark spark-two">✧</span>
                </div>

                <div className="welcome-badge">
                  <Sparkles size={13} />
                  <span>YOUR INTELLIGENT CAMPUS COMPANION</span>
                </div>

                <h1>
                  Your campus.<br />
                  <span>One conversation away.</span>
                </h1>

                <p className="welcome-description">
                  Ask questions about college regulations, academic policies
                  and student information. Upload PDFs or images to get
                  answers from your documents.
                </p>

                <div className="suggestion-grid">

                  {suggestions.map((item, index) => {
                    const Icon = item.icon

                    return (
                      <motion.button
                        key={item.title}
                        className="suggestion-card"
                        onClick={() => sendMessage(item.prompt)}
                        disabled={isLoading}
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.12 + index * 0.1 }}
                        whileHover={{ y: -5, scale: 1.015 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <div className="suggestion-icon">
                          <Icon size={19} />
                        </div>

                        <strong>{item.title}</strong>
                        <p>{item.description}</p>
                        <span className="suggestion-arrow">↗</span>
                      </motion.button>
                    )
                  })}

                </div>

              </motion.section>

            ) : (

              <section className="messages-section">

                <div className="conversation-heading">
                  <span className="conversation-tag">CONVERSATION</span>
                  <h2>{activeConversation?.title || 'Your conversation'}</h2>
                </div>


                <AnimatePresence initial={false}>

                  {messages.map(message => (

                    <motion.div
                      key={message.id}
                      className={`message-row ${message.role}`}
                      initial={{ opacity: 0, y: 14 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                    >

                      {message.role === 'assistant' && (
                        <div className="message-avatar">
                          <Sparkles size={17} />
                        </div>
                      )}

                      <div className="message-content">

                        <div className="message-author">
                          {message.role === 'user' ? 'You' : 'Campus AI'}
                        </div>

                        <div className="message-bubble">
                          {message.content}
                        </div>


                        {/* PDF CITATIONS */}

                        {message.role === 'assistant' &&
                          message.source_details &&
                          message.source_details.length > 0 && (

                          <details className="source-details">

                            <summary>
                              <FileText size={14} />
                              Sources ({message.source_details.length})
                              <ChevronDown size={14} />
                            </summary>

                            <div className="source-list">

                              {message.source_details.map((source, index) => (

                                <details
                                  className="source-item"
                                  key={`${source.filename}-${source.page}-${index}`}
                                >

                                  <summary>
                                    <FileText size={15} />

                                    <span>
                                      {source.filename} · Page {source.page}
                                    </span>
                                  </summary>

                                  <p>{source.text}</p>

                                </details>

                              ))}

                            </div>

                          </details>

                        )}


                        <button
                          className="copy-button"
                          onClick={() => copyMessage(message)}
                        >
                          {copiedId === message.id
                            ? <Check size={14} />
                            : <Copy size={14} />}

                          {copiedId === message.id ? 'Copied' : 'Copy'}
                        </button>

                      </div>

                    </motion.div>

                  ))}

                </AnimatePresence>


                {isLoading && (

                  <motion.div
                    className="message-row assistant"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >

                    <div className="message-avatar">
                      <Sparkles size={17} />
                    </div>

                    <div className="message-content">

                      <div className="message-author">Campus AI</div>

                      <div className="message-bubble">
                        Thinking... analyzing your question.
                      </div>

                    </div>

                  </motion.div>

                )}

                {/* AUTOMATIC SCROLL TARGET */}

                <div ref={messagesEndRef} />

              </section>

            )}


            {/* COMPOSER */}

            <div className="composer-area">

              <div className="composer">


                {/* IMAGE PREVIEW */}

                {selectedImage && imagePreview && (

                  <div className="attachment-preview">

                    <div className="image-preview-card">

                      <img
                        src={imagePreview}
                        alt="Selected attachment"
                      />

                      <div>
                        <strong>{selectedImage.name}</strong>
                        <span>Image ready to analyze</span>
                      </div>

                      <button
                        type="button"
                        onClick={removeImage}
                        aria-label="Remove image"
                        title="Remove image"
                      >
                        <X size={16} />
                      </button>

                    </div>

                  </div>

                )}


                <textarea
                  value={question}
                  disabled={isLoading || uploadingPdf}
                  onChange={event => setQuestion(event.target.value)}
                  onKeyDown={event => {
                    if (
                      event.key === 'Enter' &&
                      !event.shiftKey
                    ) {
                      event.preventDefault()
                      sendMessage()
                    }
                  }}
                  placeholder={
                    selectedImage
                      ? 'Ask something about this image...'
                      : 'Ask anything about your college...'
                  }
                  rows={1}
                />


                {/* UPLOAD STATUS */}

                {(uploadMessage || uploadError) && (

                  <div className={`upload-feedback ${
                    uploadError ? 'upload-error' : 'upload-success'
                  }`}>

                    {uploadError ? (
                      <X size={14} />
                    ) : (
                      <CheckCircle2 size={14} />
                    )}

                    <span>{uploadError || uploadMessage}</span>

                    <button
                      type="button"
                      onClick={() => {
                        setUploadMessage('')
                        setUploadError('')
                      }}
                      aria-label="Dismiss message"
                    >
                      <X size={14} />
                    </button>

                  </div>

                )}


                <div className="composer-footer">

                  <div className="composer-tools">

                    {/* ATTACHMENT MENU */}

                    <div className="attachment-menu-wrapper">

                      <button
                        className="tool-button"
                        type="button"
                        title="Attach a PDF or image"
                        disabled={isLoading || uploadingPdf}
                        onClick={() => {
                          setAttachmentMenuOpen(!attachmentMenuOpen)
                          setUploadError('')
                          setUploadMessage('')
                        }}
                      >
                        <Paperclip size={17} />
                      </button>


                      {attachmentMenuOpen && (

                        <div className="attachment-menu">

                          <button
                            type="button"
                            onClick={() => {
                              setAttachmentMenuOpen(false)
                              pdfInputRef.current?.click()
                            }}
                          >
                            <FileUp size={17} />
                            <span>
                              <strong>Upload PDF</strong>
                              <small>Add to permanent knowledge base</small>
                            </span>
                          </button>

                          <button
                            type="button"
                            onClick={() => {
                              setAttachmentMenuOpen(false)
                              imageInputRef.current?.click()
                            }}
                          >
                            <Image size={17} />
                            <span>
                              <strong>Upload image</strong>
                              <small>Analyze in this conversation</small>
                            </span>
                          </button>

                        </div>

                      )}

                    </div>


                    <span className="composer-hint">
                      Ask using natural language
                    </span>

                  </div>


                  <div className="composer-actions">

                    <span className="enter-hint">↵ to send</span>

                    <motion.button
                      className="send-button"
                      onClick={() => sendMessage()}
                      disabled={
                        (!question.trim() && !selectedImage) ||
                        isLoading ||
                        uploadingPdf
                      }
                      whileHover={{ scale: 1.07 }}
                      whileTap={{ scale: 0.92 }}
                      aria-label="Send message"
                    >
                      <ArrowUp size={19} />
                    </motion.button>

                  </div>

                </div>

              </div>


              <p className="disclaimer">
                <ShieldCheck size={13} />
                AI can make mistakes. Verify important information with
                official college documents.
              </p>

            </div>

          </div>

        )}


        {/* KNOWLEDGE BASE PAGE */}

        {activePage === 'knowledge' && (

          <motion.section
            className="knowledge-page"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >

            <div className="knowledge-heading">

              <div className="knowledge-heading-icon">
                <Database size={23} />
              </div>

              <div>
                <span className="conversation-tag">CAMPUSAI WORKSPACE</span>
                <h1>Knowledge Base</h1>

                <p>
                  Manage and view the documents used by your AI assistant.
                </p>
              </div>

            </div>


            <div className="knowledge-summary">

              <div className="knowledge-summary-card">
                <Database size={20} />

                <div>
                  <span>Connected documents</span>
                  <strong>{documents.length}</strong>
                </div>
              </div>

              <div className="knowledge-summary-card">
                <CheckCircle2 size={20} />

                <div>
                  <span>Knowledge status</span>
                  <strong>
                    {loadingDocuments ? 'Loading...' : 'Connected'}
                  </strong>
                </div>
              </div>

            </div>


            <div className="documents-heading">

              <div>
                <h2>Documents</h2>

                <p>
                  Documents currently indexed by your RAG system.
                </p>
              </div>

              <button
                className="refresh-documents-button"
                onClick={fetchDocuments}
                disabled={loadingDocuments}
              >
                <Upload size={15} />
                Refresh
              </button>

            </div>


            {documentError && (
              <div className="upload-feedback upload-error">
                <X size={14} />
                <span>{documentError}</span>
              </div>
            )}


            {loadingDocuments ? (

              <div className="empty-documents">
                Loading documents...
              </div>

            ) : documents.length > 0 ? (

              <div className="documents-list">

                {documents.map(document => (

                  <div
                    className="document-card"
                    key={document.filename}
                  >

                    <div className="document-icon">
                      <FileText size={25} />
                    </div>

                    <div className="document-info">

                      <h3>{document.filename}</h3>

                      <p>
                        PDF document · {document.indexed_pages} indexed pages
                      </p>

                      <div className="document-status">
                        <CheckCircle2 size={14} />
                        <span>Connected to RAG system</span>
                      </div>

                    </div>

                    <div className="document-badge">
                      <span className="status-dot" />
                      Active
                    </div>

                  </div>

                ))}

              </div>

            ) : (

              <div className="empty-documents">
                No indexed documents found. Upload a PDF from the chat
                attachment button.
              </div>

            )}


            <div className="knowledge-info">

              <div className="knowledge-info-icon">
                <Sparkles size={19} />
              </div>

              <div>
                <h3>How your knowledge base works</h3>

                <p>
                  When you ask a question, CampusAI searches indexed
                  college PDFs using FAISS and BM25, retrieves relevant
                  text, and sends that context to Groq to generate an answer.
                  Images are analyzed separately and are not indexed.
                </p>
              </div>

            </div>


            <div className="knowledge-note">
              <Clock size={16} />

              <p>
                Uploaded PDFs remain in the knowledge base after
                restarting the backend. They are shared across users
                of this prototype.
              </p>
            </div>

          </motion.section>

        )}


        {/* FOOTER */}

        <footer className="app-footer">

          <span>
            CampusAI <span className="footer-dot">·</span> College knowledge assistant
          </span>

          <span>Powered by intelligent retrieval</span>

        </footer>

      </main>

    </div>
  )
}

export default App
